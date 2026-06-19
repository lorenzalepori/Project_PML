import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from Training_model import (build_windows, schedule, ConditionalDenoiser,
    SPLITS, PROCESSED_DIR, PAST_FEATS, FUT_FEATS, TARGETS, OUTPUT_LEN, DEVICE,)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SAMPLES      = 40
GUIDANCE       = 0.0
BASE_SEED      = 1234
CLAMP_Z        = 3.0
LOG_CAP        = 13.0
MODIFY_PAST    = True

VAX_IDX_PAST = PAST_FEATS.index("vaccine")
MOB_IDX_PAST = PAST_FEATS.index("mobility")

SCENARIOS = {
    "real_case":      dict(vaccine_factor=1.0, mobility_factor=0.3),
    "no_intervention":     dict(vaccine_factor=0.0, mobility_factor=1.0),
    "no_vax_restrictions": dict(vaccine_factor=0.0, mobility_factor=0.6),
    "no_restrictions_vax": dict(vaccine_factor=0.6, mobility_factor=1.0),
}

COLORS = {
    "real_case":      "steelblue",
    "no_intervention":     "orange",
    "no_vax_restrictions": "green",
    "no_restrictions_vax": "crimson",
}


def load_norm_stats():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "norm_stats.csv"))
    return {(r["region"], r["variable"]): (float(r["mu"]), float(r["sigma"]))
            for _, r in df.iterrows()}


def target_to_abs(z, region, stats, pop):
    """z-score di log1p(tasso/100k) -> conteggi giornalieri assoluti."""
    out = np.empty_like(z, dtype=np.float64)
    for j, var in enumerate(TARGETS):
        mu, sigma = stats[(region, var)]
        arg = np.clip(z[..., j] * sigma + mu, None, LOG_CAP)
        rate = np.clip(np.expm1(arg), 0, None)
        out[..., j] = rate * pop / 1e5
    return out


def scale_var(arr, feat_list, idx_map, region, stats):
    out = arr.clone()
    for var, (j, factor) in idx_map.items():
        mu, sigma = stats[(region, var)]
        out[..., j] = factor * out[..., j] + (factor - 1) * (mu / sigma)
    return out


@torch.no_grad()
def ddpm_sample(model, sched, x_past, c_fut, rid, n_samples,
                guidance=0.0, base_seed=0):
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
            y = y.clamp(-CLAMP_Z, CLAMP_Z)
        samples.append(y.cpu().numpy())
    return np.stack(samples)


def finite(a):
    a = np.asarray(a, dtype=np.float64)
    return a[np.isfinite(a)]


def main():
    ckpt = torch.load(os.path.join(PROCESSED_DIR, "diffusion_ckpt.pt"),
                      map_location=DEVICE)
    cfg = ckpt["config"]
    model = ConditionalDenoiser(n_targets=len(TARGETS), n_fut=len(FUT_FEATS),
                                n_past=len(PAST_FEATS)).to(DEVICE)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    sched = schedule(cfg["T_STEPS"], cfg["BETA_START"], cfg["BETA_END"], DEVICE)
    stats = load_norm_stats()

    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_panel.csv"),
                        parse_dates=["date"])
    id2reg = (panel[["region_id", "region"]].drop_duplicates()
              .set_index("region_id")["region"].to_dict())
    pop_by_reg = (panel.groupby("region")["pop_log"].first()
                  .apply(np.exp).to_dict())

    s, e = SPLITS["test"]
    x_past, c_fut, y_true, rid = build_windows(panel, s, e)

    keep = []
    for r in np.unique(rid):
        idx = np.where(rid == r)[0]
        keep.extend(idx[::OUTPUT_LEN].tolist())
    keep = np.array(sorted(keep))
    x_past, c_fut, rid = x_past[keep], c_fut[keep], rid[keep]
    print(f"Test: {len(keep)} non overlapping windows x {N_SAMPLES} samples "
          f"x {len(SCENARIOS)} scenarios")

    x_past_t = torch.from_numpy(x_past).to(DEVICE)
    c_fut_t  = torch.from_numpy(c_fut).to(DEVICE)
    rid_t    = torch.from_numpy(rid).to(DEVICE)

    scen_abs = {}
    for name, fac in SCENARIOS.items():
        c_mod = c_fut_t.clone()
        x_mod = x_past_t.clone()
        for i in range(len(rid)):
            reg = id2reg[int(rid[i])]
            fut_map = {"vaccine": (FUT_FEATS.index("vaccine"), fac["vaccine_factor"]),
                       "mobility": (FUT_FEATS.index("mobility"), fac["mobility_factor"])}
            c_mod[i] = scale_var(c_fut_t[i], FUT_FEATS, fut_map, reg, stats)
            if MODIFY_PAST:
                past_map = {"vaccine": (VAX_IDX_PAST, fac["vaccine_factor"]),
                            "mobility": (MOB_IDX_PAST, fac["mobility_factor"])}
                x_mod[i] = scale_var(x_past_t[i], PAST_FEATS, past_map, reg, stats)

        s_z = ddpm_sample(model, sched, x_mod, c_mod, rid_t,
                          N_SAMPLES, GUIDANCE, base_seed=BASE_SEED)

        p = np.percentile(s_z, [0, 1, 50, 99, 100])
        print(f"  [diagnostic {name}] z: min={p[0]:.2f} p1={p[1]:.2f} "
              f"mediana={p[2]:.2f} p99={p[3]:.2f} max={p[4]:.2f} "
              f"media={s_z.mean():.3f} std={s_z.std():.3f}")

        s_abs = np.empty_like(s_z, dtype=np.float64)
        for i in range(s_z.shape[1]):
            reg = id2reg[int(rid[i])]
            s_abs[:, i] = target_to_abs(s_z[:, i], reg, stats, pop_by_reg[reg])
        scen_abs[name] = s_abs

    base = scen_abs["real_case"]
    rows = []
    per_draw_deaths = {}
    for name, samp in scen_abs.items():
        tot_cases  = np.nansum(samp[..., 0], axis=(1, 2))
        tot_deaths = np.nansum(samp[..., 1], axis=(1, 2))
        d_cases    = np.nansum(samp[..., 0] - base[..., 0], axis=(1, 2))
        d_deaths   = np.nansum(samp[..., 1] - base[..., 1], axis=(1, 2))
        peak       = np.nanmean(np.nanmax(samp[..., 0], axis=2), axis=1)
        per_draw_deaths[name] = finite(tot_deaths)
        rows.append({
            "scenario": name,
            "casi_tot_mediana":      np.median(finite(tot_cases)),
            "casi_tot_p05":          np.percentile(finite(tot_cases), 5),
            "casi_tot_p95":          np.percentile(finite(tot_cases), 95),
            "decessi_tot_mediana":   np.median(finite(tot_deaths)),
            "decessi_tot_p05":       np.percentile(finite(tot_deaths), 5),
            "decessi_tot_p95":       np.percentile(finite(tot_deaths), 95),
            "picco_casi_mediana":    np.median(finite(peak)),
            "casi_vs_S1_mediana":    np.median(finite(d_cases)),
            "decessi_vs_S1_mediana": np.median(finite(d_deaths)),
            "decessi_vs_S1_p05":     np.percentile(finite(d_deaths), 5),
            "decessi_vs_S1_p95":     np.percentile(finite(d_deaths), 95),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUTPUT_DIR, "scenari_risultati.csv"), index=False)
    print(table.to_string(index=False))

    def col(scn, c):
        return table.loc[table.scenario == scn, c].iloc[0]

    print("\nDeaths avoided by the vaccine:"
          f"   mediana {col('no_intervention','decessi_vs_S1_mediana'):,.0f}"
          f"   [{col('no_intervention','decessi_vs_S1_p05'):,.0f},"
          f" {col('no_intervention','decessi_vs_S1_p95'):,.0f}]")
    print("Additional effect of mobility restriction (no_restrictions_vax - no_intervention): ")

    #Trajectories graphs
    plt.figure(figsize=(9, 5))
    for name, samp in scen_abs.items():
        traj = np.nanmean(samp[:, :, :, 1], axis=(0, 1))
        plt.plot(range(OUTPUT_LEN), traj, label=name, color=COLORS[name], linewidth=2)
    plt.xlabel("Day")
    plt.ylabel("Average Daily Deaths")
    plt.title("Counterfactual Trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deaths_trajectories.png"), dpi=130)
    plt.close()

    #KDE graph distribution of total deaths
    plt.figure(figsize=(9, 5))
    for name, vals in per_draw_deaths.items():
        if vals.size:
            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 300)
            plt.plot(x, kde(x), label=name, color=COLORS[name], linewidth=2)
            plt.axvline(np.median(vals), color=COLORS[name], linestyle="--", alpha=0.6)
    plt.xlabel("Total deaths over 30 days")
    plt.ylabel("Density")
    plt.title("Deaths distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "deaths_distribution.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()