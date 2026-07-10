import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DIRECTORY = os.path.dirname(os.path.abspath(__file__))

def _find_processed_dir():
    cand = os.path.join(DIRECTORY, "processed_data")
    if os.path.exists(os.path.join(cand, "dataset_panel.csv")):
        return cand
    root = os.path.dirname(DIRECTORY)
    for dirpath, _, files in os.walk(root):
        if "dataset_panel.csv" in files:
            return dirpath
    raise FileNotFoundError(
        f"dataset_panel.csv not found: run data_preparation.py first. "
        f"Searched in {cand} and subfolders of {root}."
    )

PROCESSED_DIR = _find_processed_dir()

INPUT_LEN, OUTPUT_LEN = 60, 30

# Past features
PAST_FEATS = ["cases", "deaths", "vaccine_elderly", "vaccine_young", "mobility",
              "season_sin", "season_cos", "pop_log", "region_id"]

# Future covariates
FUT_FEATS  = ["vaccine_elderly", "vaccine_young", "mobility",
              "season_sin", "season_cos"]

TARGETS   = ["cases", "deaths"]
N_REGIONS = 20

START = pd.Timestamp("2021-01-01")
END   = pd.Timestamp("2022-12-31")

# Temporal split
SPLITS = {
    "train": (START,                        pd.Timestamp("2021-09-30")),
    "val":   (pd.Timestamp("2021-10-01"),   pd.Timestamp("2021-12-31")),
    "test":  (pd.Timestamp("2022-01-01"),   pd.Timestamp("2022-03-31")),
}

# Hyperparameters
T_STEPS      = 500
BETA_START   = 1e-4
BETA_END     = 0.02
EPOCHS       = 100
BATCH_SIZE   = 128
LR           = 1.5e-4
WIDTH        = 24
COND_DIM     = 48
COND_DROPOUT = 0.4
EMA_DECAY    = 0.999
SEED         = 0
DEVICE       = "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_windows(panel, period_start, period_end):
    """Build windows whose 30-day target falls within [period_start, period_end]."""
    X_past, C_fut, Y, RID = [], [], [], []
    for region, g in panel.groupby("region"):
        g    = g.sort_values("date").reset_index(drop=True)
        past = g[PAST_FEATS].to_numpy(np.float32)
        fut  = g[FUT_FEATS].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        rid  = int(g["region_id"].iloc[0])
        for t in range(INPUT_LEN, len(g) - OUTPUT_LEN + 1):
            tgt_start = g["date"].iloc[t]
            tgt_end   = g["date"].iloc[t + OUTPUT_LEN - 1]
            if tgt_start >= period_start and tgt_end <= period_end:
                X_past.append(past[t - INPUT_LEN:t])
                C_fut.append(fut[t:t + OUTPUT_LEN])
                Y.append(targ[t:t + OUTPUT_LEN])
                RID.append(rid)
    return (np.stack(X_past), np.stack(C_fut), np.stack(Y),
            np.array(RID, dtype=np.int64))


class WindowDataset(Dataset):
    def __init__(self, x_past, c_fut, y, rid):
        self.x_past = torch.from_numpy(x_past)
        self.c_fut  = torch.from_numpy(c_fut)
        self.y      = torch.from_numpy(y)
        self.rid    = torch.tensor(rid)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_past[idx], self.c_fut[idx], self.y[idx], self.rid[idx]


def schedule(t, b0, b1, device):
    """Linear noise schedule from b0 to b1 over t steps."""
    betas  = torch.linspace(b0, b1, t, device=device)
    alphas = 1.0 - betas
    abar   = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas, "alphas": alphas, "abar": abar,
        "sqrt_abar": torch.sqrt(abar),
        "sqrt_one_minus_abar": torch.sqrt(1.0 - abar)
    }


def timestep_embedding(t, dim):
    """Sinusoidal timestep embedding."""
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args  = t[:, None].float() * freqs[None]
    emb   = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class PastEncoder(nn.Module):
    """Encodes 60-day past context into a single conditioning vector."""
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(32, 64, 5, padding=2),    nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(64, 64, 3, padding=1),    nn.GELU(),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x_past):
        h = self.net(x_past.transpose(1, 2))  # (B, 64, 60)
        h = h.mean(dim=-1)                     # global average pooling → (B, 64)
        return self.head(h)


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning and dilated convolution."""
    def __init__(self, ch, cond_dim, fut_ch, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch + fut_ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film  = nn.Linear(cond_dim, 2 * ch)

    def forward(self, h, cond, fut):
        scale, shift = self.film(cond)[..., None].chunk(2, dim=1)
        x = self.norm1(h) * (1 + scale) + shift
        x = F.gelu(x)
        x = self.conv1(torch.cat([x, fut], dim=1))
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return h + x


class ConditionalDenoiser(nn.Module):
    """Conditional DDPM denoiser."""
    def __init__(self, n_targets=2, n_fut=5, n_past=9,
                 width=WIDTH, cond_dim=COND_DIM, n_regions=N_REGIONS):
        super().__init__()
        self.cond_dim   = cond_dim
        self.past_enc   = PastEncoder(n_past, cond_dim)
        self.region_emb = nn.Embedding(n_regions, cond_dim)
        self.null_cond  = nn.Parameter(torch.zeros(cond_dim))
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.fut_proj = nn.Conv1d(n_fut, width, 1)
        self.in_conv  = nn.Conv1d(n_targets, width, 1)
        dilations = [1, 2, 4, 1, 2, 4]
        self.blocks = nn.ModuleList(
            [ResBlock(width, cond_dim, width, d) for d in dilations])
        self.out = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, n_targets, 1),
        )

    def forward(self, y_t, t, x_past, c_fut, rid, drop_mask=None):
        ctx = self.past_enc(x_past) + self.region_emb(rid)
        fut = self.fut_proj(c_fut.transpose(1, 2))
        if drop_mask is not None:
            m   = drop_mask.float()[:, None]
            ctx = torch.where(drop_mask[:, None], self.null_cond[None], ctx)
            fut = fut * (1 - m[..., None])
        cond = ctx + self.t_mlp(timestep_embedding(t, self.cond_dim))
        h    = self.in_conv(y_t.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h, cond, fut)
        return self.out(h).transpose(1, 2)  # (B, 30, n_targets)


class EMA:
    def __init__(self, model, decay):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone()


def diffusion_loss(model, sched, batch):
    x_past, c_fut, y0, rid = [b.to(DEVICE) for b in batch]
    B    = y0.shape[0]
    t    = torch.randint(0, T_STEPS, (B,), device=DEVICE)
    eps  = torch.randn_like(y0)
    sa   = sched["sqrt_abar"][t][:, None, None]
    soma = sched["sqrt_one_minus_abar"][t][:, None, None]
    y_t  = sa * y0 + soma * eps
    drop = (torch.rand(B, device=DEVICE) < COND_DROPOUT) if COND_DROPOUT > 0 else None
    eps_pred = model(y_t, t, x_past, c_fut, rid, drop_mask=drop)
    return F.mse_loss(eps_pred, eps)


def main():
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_panel.csv"),
                        parse_dates=["date"])

    data = {}
    for name, (s, e) in SPLITS.items():
        data[name] = build_windows(panel, s, e)
        print(f"{name}: {len(data[name][2])} windows")

    train_ds = WindowDataset(*data["train"])
    val_ds   = WindowDataset(*data["val"])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    sched = schedule(T_STEPS, BETA_START, BETA_END, DEVICE)
    model = ConditionalDenoiser(n_targets=len(TARGETS),
                                n_fut=len(FUT_FEATS),
                                n_past=len(PAST_FEATS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    ema = EMA(model, EMA_DECAY)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr = 0.0
        for batch in train_dl:
            opt.zero_grad()
            loss = diffusion_loss(model, sched, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            tr += loss.item() * batch[0].size(0)
        tr /= len(train_ds)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_dl:
                vl += diffusion_loss(model, sched, batch).item() * batch[0].size(0)
        vl /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {tr:.4f} | val {vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": model.state_dict(),
                 "ema":   ema.shadow,
                 "config": {"T_STEPS": T_STEPS, "BETA_START": BETA_START,
                            "BETA_END": BETA_END, "WIDTH": WIDTH,
                            "COND_DIM": COND_DIM, "PAST_FEATS": PAST_FEATS,
                            "FUT_FEATS": FUT_FEATS, "TARGETS": TARGETS}},
                os.path.join(PROCESSED_DIR, "diffusion_ckpt.pt"))

    print(f"Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()