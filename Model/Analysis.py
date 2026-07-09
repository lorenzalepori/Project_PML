import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.ndimage import uniform_filter1d

from Training_model import (build_windows, schedule, ConditionalDenoiser,
    SPLITS, PROCESSED_DIR, PAST_FEATS, FUT_FEATS, TARGETS, OUTPUT_LEN,
    INPUT_LEN, DEVICE, N_REGIONS)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Configuration ---
N_SAMPLES    = 40
N_WINDOWS    = 3       # autoregressive rollout: 3 x 30 = 90 days total
GUIDANCE     = 0.5
BASE_SEED    = 1234
LOG_CAP      = 13.0

# Feature indices
VAX_ELD_IDX_PAST = PAST_FEATS.index("vaccine_elderly")
VAX_YNG_IDX_PAST = PAST_FEATS.index("vaccine_young")
MOB_IDX_PAST     = PAST_FEATS.index("mobility")
SEAS_SIN_IDX_PAST = PAST_FEATS.index("season_sin")
SEAS_COS_IDX_PAST = PAST_FEATS.index("season_cos")
POP_LOG_IDX_PAST  = PAST_FEATS.index("pop_log")
RID_IDX_PAST      = PAST_FEATS.index("region_id")
CASES_IDX_PAST    = PAST_FEATS.index("cases")
DEATHS_IDX_PAST   = PAST_FEATS.index("deaths")

VAX_ELD_IDX_FUT  = FUT_FEATS.index("vaccine_elderly")
VAX_YNG_IDX_FUT  = FUT_FEATS.index("vaccine_young")
MOB_IDX_FUT      = FUT_FEATS.index("mobility")
SEAS_SIN_IDX_FUT = FUT_FEATS.index("season_sin")
SEAS_COS_IDX_FUT = FUT_FEATS.index("season_cos")

# Scenarios:
# vaccine_factor: multiplier on raw vaccine coverage (1.0 = observed, 0.0 = no vaccines)
# mobility_factor: multiplier on raw mobility (1.0 = observed)
SCENARIOS = {
    "real_case":           dict(vaccine_factor=1.0, mobility_factor=1.0),
    "no_intervention":     dict(vaccine_factor=0.0, mobility_factor=1.0),
    "no_vax_restrictions": dict(vaccine_factor=0.0, mobility_factor=0.6),
    "no_restrictions_vax": dict(vaccine_factor=0.2, mobility_factor=1.0),
}

COLORS = {
    "real_case":           "steelblue",
    "no_intervention":     "orange",
    "no_vax_restrictions": "green",
    "no_restrictions_vax": "crimson",
}

TOTAL_DAYS = N_WINDOWS * OUTPUT_LEN  # 90 days


# --- Helper functions ---

def load_norm_stats():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "norm_stats.csv"))
    return {(r["region"], r["variable"]): (float(r["mu"]), float(r["sigma"]))
            for _, r in df.iterrows()}


def target_to_abs(z, region, stats, pop):
    """Convert z-scores of log1p(rate per 100k) to absolute daily counts."""
    out = np.empty_like(z, dtype=np.float64)
    for j, var in enumerate(TARGETS):
        mu, sigma = stats[(region, var)]
        arg = np.clip(z[..., j] * sigma + mu, None, LOG_CAP)
        rate = np.clip(np.expm1(arg), 0, None)
        out[..., j] = rate * pop / 1e5
    return out


def scale_raw(arr, j, factor, var, region, stats):
    """Scale covariate j: denormalize → multiply by factor → renormalize."""
    out = arr.clone()
    mu, sigma = stats[(region, var)]
    raw = out[..., j] * sigma + mu
    out[..., j] = (raw * factor - mu) / sigma
    return out


def get_panel_slice(panel, region, date_start, n_days):
    """Extract n_days rows from panel for a given region starting at date_start."""
    g = panel[panel["region"] == region].sort_values("date").reset_index(drop=True)
    idx = g[g["date"] >= date_start].index
    if len(idx) == 0:
        return None
    start_i = idx[0]
    return g.iloc[start_i:start_i + n_days]


@torch.no_grad()
def ddpm_sample(model, sched, x_past, c_fut, rid, n_samples,
                guidance=0.0, base_seed=0):
    """Ancestral DDPM sampling."""
    B = x_past.shape[0]
    betas, alphas, abar = sched["betas"], sched["alphas"], sched["abar"]
    T = len(betas)
    samples = []
    for i in range(n_samples):
        g = torch.Generator(device=DEVICE)
        g.manual_seed(base_seed + i)
        y = torch.randn(B, OUTPUT_LEN, len(TARGETS), device=DEVICE, generator=g)
        for t in reversed(range(T)):
            tt = torch.full((B,), t, device=DEVICE, dtype=torch.long)
            eps = model(y, tt, x_past, c_fut, rid)
            if guidance > 0:
                drop = torch.ones(B, dtype=torch.bool, device=DEVICE)
                eps_u = model(y, tt, x_past, c_fut, rid, drop_mask=drop)
                eps = (1 + guidance) * eps - guidance * eps_u
            a_t, ab_t = alphas[t], abar[t]
            mean = (y - (1 - a_t) / torch.sqrt(1 - ab_t) * eps) / torch.sqrt(a_t)
            if t > 0:
                noise = torch.randn(y.shape, device=DEVICE, generator=g)
                y = mean + torch.sqrt(betas[t]) * noise
            else:
                y = mean
        samples.append(y.cpu().numpy())
    return np.stack(samples)  # (n_samples, B, 30, 2)


def build_future_covariates(panel, region, date_start, n_days, fac, stats):
    """
    Build future covariate tensor (n_days, n_fut) for a given scenario.
    - season_sin, season_cos: from real panel data
    - vaccine_elderly, vaccine_young, mobility: scaled according to scenario
    """
    slice_df = get_panel_slice(panel, region, date_start, n_days)
    c = np.zeros((n_days, len(FUT_FEATS)), dtype=np.float32)

    if slice_df is not None and len(slice_df) >= n_days:
        for j, feat in enumerate(FUT_FEATS):
            c[:, j] = slice_df[feat].values[:n_days]
    else:
        # fallback: zeros (should not happen in test period)
        return torch.from_numpy(c)

    c_t = torch.from_numpy(c)

    # Scale vaccine and mobility according to scenario
    vf = fac["vaccine_factor"]
    mf = fac["mobility_factor"]
    if vf != 1.0:
        c_t = scale_raw(c_t, VAX_ELD_IDX_FUT, vf, "vaccine_elderly", region, stats)
        c_t = scale_raw(c_t, VAX_YNG_IDX_FUT, vf, "vaccine_young",   region, stats)
    if mf != 1.0:
        c_t = scale_raw(c_t, MOB_IDX_FUT, mf, "mobility", region, stats)

    return c_t  # (n_days, n_fut)


def build_past_context_from_generated(
        prev_context,        # (60, n_past) tensor — previous context
        generated_median,    # (30, 2) numpy — median of generated cases/deaths in z-score
        panel, region,
        date_start,          # start date of the generated window
        fac, stats):
    """
    Build new 60-day past context for the next rollout window:
    - cases, deaths: from generated median (z-score)
    - vaccine_elderly, vaccine_young, mobility: from real panel, scaled by scenario
    - season_sin, season_cos, pop_log, region_id: from real panel
    """
    # The new context is: last 30 days of prev_context + 30 new days
    new_context = torch.zeros(INPUT_LEN, len(PAST_FEATS), dtype=torch.float32)

    # First 30 days: take last 30 from previous context
    new_context[:30] = prev_context[30:]

    # Last 30 days: build from generated + real panel
    slice_df = get_panel_slice(panel, region, date_start, OUTPUT_LEN)

    for j, feat in enumerate(PAST_FEATS):
        if feat == "cases":
            new_context[30:, j] = torch.from_numpy(generated_median[:, 0].astype(np.float32))
        elif feat == "deaths":
            new_context[30:, j] = torch.from_numpy(generated_median[:, 1].astype(np.float32))
        elif slice_df is not None and len(slice_df) >= OUTPUT_LEN:
            vals = slice_df[feat].values[:OUTPUT_LEN].astype(np.float32)
            new_context[30:, j] = torch.from_numpy(vals)

    # Scale vaccine and mobility in the new 30-day block according to scenario
    vf = fac["vaccine_factor"]
    mf = fac["mobility_factor"]
    if vf != 1.0:
        new_context[30:] = scale_raw(new_context[30:], VAX_ELD_IDX_PAST, vf,
                                      "vaccine_elderly", region, stats)
        new_context[30:] = scale_raw(new_context[30:], VAX_YNG_IDX_PAST, vf,
                                      "vaccine_young",   region, stats)
    if mf != 1.0:
        new_context[30:] = scale_raw(new_context[30:], MOB_IDX_PAST, mf,
                                      "mobility", region, stats)

    return new_context  # (60, n_past)


def finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


# --- Main ---

def main():
    # Load model
    ckpt = torch.load(os.path.join(PROCESSED_DIR, "diffusion_ckpt.pt"),
                      map_location=DEVICE)
    cfg = ckpt["config"]
    model = ConditionalDenoiser(n_targets=len(TARGETS), n_fut=len(FUT_FEATS),
                                n_past=len(PAST_FEATS)).to(DEVICE)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    sched = schedule(cfg["T_STEPS"], cfg["BETA_START"], cfg["BETA_END"], DEVICE)
    stats = load_norm_stats()

    # Load panel
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_panel.csv"),
                        parse_dates=["date"])
    id2reg = (panel[["region_id", "region"]].drop_duplicates()
              .set_index("region_id")["region"].to_dict())
    pop_by_reg = (panel.groupby("region")["pop_log"].first()
                  .apply(np.exp).to_dict())

    # Build non-overlapping test windows (first window only — rollout handles the rest)
    s_test, e_test = SPLITS["test"]
    x_past_all, c_fut_all, y_true_all, rid_all = build_windows(panel, s_test, e_test)

    # Keep only the first non-overlapping window per region
    keep = []
    for r in np.unique(rid_all):
        idx = np.where(rid_all == r)[0]
        keep.append(idx[0])  # first window only — rollout generates the rest
    keep = np.array(sorted(keep))

    x_past_init = x_past_all[keep]   # (B, 60, n_past)
    c_fut_init  = c_fut_all[keep]    # (B, 30, n_fut) — first window future covariates
    rid         = rid_all[keep]

    # Get start dates for each window
    # Window 1 starts at s_test, window 2 at s_test+30, window 3 at s_test+60
    window_starts = [s_test + pd.Timedelta(days=w * OUTPUT_LEN) for w in range(N_WINDOWS)]

    B = len(keep)
    print(f"Autoregressive rollout: {B} regions x {N_WINDOWS} windows x "
          f"{N_SAMPLES} samples x {len(SCENARIOS)} scenarios")
    print(f"Total horizon: {TOTAL_DAYS} days\n")

    # --- Autoregressive rollout for each scenario ---
    # scen_abs[name] shape: (n_samples, B, TOTAL_DAYS, 2)
    scen_abs = {}

    for name, fac in SCENARIOS.items():
        print(f"Scenario: {name}")
        all_windows_abs = []   # list of n_windows arrays, each (n_samples, B, 30, 2)
        all_windows_z   = []   # same but in z-score space

        # Initialize context with real observed past
        contexts = [torch.from_numpy(x_past_init[i]).clone()
                    for i in range(B)]  # list of B tensors (60, n_past)

        for w in range(N_WINDOWS):
            w_start = window_starts[w]
            print(f"  Window {w+1}/3 starting {w_start.date()}")

            # Build future covariates for this window, per region
            c_fut_w = torch.zeros(B, OUTPUT_LEN, len(FUT_FEATS))
            for i in range(B):
                reg = id2reg[int(rid[i])]
                c_fut_w[i] = build_future_covariates(
                    panel, reg, w_start, OUTPUT_LEN, fac, stats)

            # Stack contexts
            x_past_t = torch.stack(contexts).to(DEVICE)   # (B, 60, n_past)
            c_fut_t  = c_fut_w.to(DEVICE)
            rid_t    = torch.from_numpy(rid).to(DEVICE)

            # Sample
            s_z = ddpm_sample(model, sched, x_past_t, c_fut_t, rid_t,
                               N_SAMPLES, GUIDANCE,
                               base_seed=BASE_SEED + w * 10000)
            # s_z shape: (n_samples, B, 30, 2)

            # Diagnostics
            p = np.percentile(s_z, [1, 50, 99])
            print(f"    z-score: p1={p[0]:.2f} median={p[1]:.2f} p99={p[2]:.2f} "
                  f"mean={s_z.mean():.3f} std={s_z.std():.3f}")

            # Back-transform to absolute counts
            s_abs = np.empty_like(s_z, dtype=np.float64)
            for i in range(B):
                reg = id2reg[int(rid[i])]
                s_abs[:, i] = target_to_abs(s_z[:, i], reg, stats, pop_by_reg[reg])

            all_windows_abs.append(s_abs)
            all_windows_z.append(s_z)

            # Update contexts for next window using median of generated z-scores
            if w < N_WINDOWS - 1:
                mean_z = np.mean(s_z, axis=0)  # (B, 30, 2)
                for i in range(B):
                    reg = id2reg[int(rid[i])]
                    next_start = window_starts[w + 1]
                    contexts[i] = build_past_context_from_generated(
                        contexts[i], mean_z[i],
                        panel, reg, next_start, fac, stats)

        # Concatenate windows along time axis → (n_samples, B, TOTAL_DAYS, 2)
        scen_abs[name] = np.concatenate(all_windows_abs, axis=2)
        print(f"  {name}: done | mean deaths={scen_abs[name][...,1].mean():.2f}\n")

    # --- Back-transform observed data (90 days) ---
    real_abs_mat = np.zeros((B, TOTAL_DAYS))
    for i in range(B):
        reg = id2reg[int(rid[i])]
        mu, sigma = stats[(reg, "deaths")]
        for w in range(N_WINDOWS):
            w_start = window_starts[w]
            slice_df = get_panel_slice(panel, reg, w_start, OUTPUT_LEN)
            if slice_df is not None and len(slice_df) >= OUTPUT_LEN:
                z_obs = slice_df["deaths"].values[:OUTPUT_LEN]
                arg = np.clip(z_obs * sigma + mu, None, LOG_CAP)
                real_abs_mat[i, w*OUTPUT_LEN:(w+1)*OUTPUT_LEN] = (
                    np.clip(np.expm1(arg), 0, None) * pop_by_reg[reg] / 1e5)

    real_traj = np.nanmean(real_abs_mat, axis=0)  # (TOTAL_DAYS,)
    real_traj_smooth = uniform_filter1d(real_traj, size=5)
    total_observed = np.nansum(real_abs_mat)
    print(f"Total observed deaths ({TOTAL_DAYS} days): {total_observed:.0f}")

    # --- MAE calibration (real_case vs observed) ---
    pred_real = np.nanmean(scen_abs["real_case"][..., 1], axis=0)  # (B, TOTAL_DAYS)
    mae_global = np.nanmean(np.abs(pred_real - real_abs_mat))
    print(f"MAE deaths (real_case vs observed, {TOTAL_DAYS} days): {mae_global:.4f}\n")

    # --- Per-draw totals (for boxplot) ---
    per_draw_deaths = {name: finite(np.nansum(samp[..., 1], axis=(1, 2)))
                        for name, samp in scen_abs.items()}

    # --- Parallel trends check ---
    print("\n=== Parallel trends check ===")
    base_traj = np.nanmean(scen_abs["real_case"][:, :, :, 1], axis=(0, 1))
    for name, samp in scen_abs.items():
        if name == "real_case":
            continue
        traj = np.nanmean(samp[:, :, :, 1], axis=(0, 1))
        diff = traj - base_traj
        print(f"  {name}: mean diff={diff.mean():.2f}  std={diff.std():.2f}")

    days = np.arange(TOTAL_DAYS)
    # Vertical lines to mark window boundaries
    window_lines = [OUTPUT_LEN, 2 * OUTPUT_LEN]

    # --- Plot 1: National trajectories vs Observed ---
    plt.figure(figsize=(11, 5))
    for name, samp in scen_abs.items():
        traj = uniform_filter1d(np.nanmean(samp[:, :, :, 1], axis=(0, 1)), size=5)
        plt.plot(days, traj, label=name, color=COLORS[name], linewidth=2)
    plt.plot(days, real_traj_smooth, label="Observed",
             color="black", linewidth=2, linestyle="--")
    for xl in window_lines:
        plt.axvline(xl, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    plt.xlabel(f"Day (0–{TOTAL_DAYS}, 3 autoregressive windows of 30 days)")
    plt.ylabel("Mean daily deaths")
    plt.title("Counterfactual Trajectories vs Observed — All regions (90-day horizon)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deaths_trajectories_national.png"), dpi=130)
    plt.close()

    # --- Plot 2: Boxplot (Observed shown as a box, replacing real_case) ---
    plot_names = [n for n in SCENARIOS.keys() if n != "real_case"]
    observed_per_region = np.nansum(real_abs_mat, axis=1)  # (B,) totale decessi osservati per regione

    plt.figure(figsize=(9, 5))
    data_to_plot = [per_draw_deaths[name] for name in plot_names]
    data_to_plot.append(finite(observed_per_region))
    labels = plot_names + ["Observed"]
    box_colors = [COLORS[name] for name in plot_names] + ["gray"]

    bp = plt.boxplot(data_to_plot, tick_labels=labels,
                     patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    plt.ylabel(f"Total deaths ({TOTAL_DAYS}-day horizon)")
    plt.title("Distribution of Total Deaths by Scenario")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deaths_boxplot.png"), dpi=130)
    plt.close()

    # --- Plot 3: Window-by-window deaths (stacked bar), real_case excluded ---
    plot_names = [n for n in SCENARIOS.keys() if n != "real_case"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(plot_names))
    width = 0.6
    bottoms = np.zeros(len(plot_names))
    window_colors = ["#4a90d9", "#7bb8f5", "#b8d9f7"]
    for w in range(N_WINDOWS):
        medians = []
        for name in plot_names:
            samp = scen_abs[name]
            win_deaths = np.nansum(samp[:, :, w*OUTPUT_LEN:(w+1)*OUTPUT_LEN, 1],
                                   axis=(1, 2))
            medians.append(np.median(finite(win_deaths)))
        medians = np.array(medians)
        ax.bar(x, medians, width, bottom=bottoms,
               color=window_colors[w], label=f"Window {w+1} (days {w*30+1}-{(w+1)*30})",
               alpha=0.85)
        bottoms += medians
    ax.axhline(total_observed, color="black", linewidth=2,
               linestyle="--", label="Observed total")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_names, rotation=15)
    ax.set_ylabel("Total deaths (median)")
    ax.set_title("Deaths by Window and Scenario")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deaths_by_window.png"), dpi=130)
    plt.close()

    print(f"\nAll plots and tables saved in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()