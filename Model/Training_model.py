import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#Learns p(cases, deaths | vaccine, mobility, region, season) 
#The model in the end will generate the target variables Y=(deaths, cases) on the following 30 days
#For the past we input the state of the epidemy: (deaths, cases, vaccines, mobility, season, population log)
#For the future we only input (vaccines, mobility, season) which are the ones we want to control
#Here we perform the forward pass -> injecting Gaussian noise


DIRECTORY      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR  = os.path.join(DIRECTORY, "processed_data")
INPUT_LEN, OUTPUT_LEN = 60, 30   #60 days of past data, 30 days of future data
PAST_FEATS = ["cases", "deaths", "vaccine", "mobility", "season_sin", "season_cos", "pop_log"]
FUT_FEATS  = ["vaccine", "mobility", "season_sin", "season_cos"] #pop is constant so no need to include it in the future features
TARGETS    = ["cases", "deaths"]
N_REGIONS = 20

START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2022-12-31")

#Temporal split: no random shuffling, we want to predict the future
SPLITS ={
    "train": (START, pd.Timestamp("2022-06-30")),
    "val": (pd.Timestamp("2022-07-01"), pd.Timestamp("2022-09-30")),
    "test": (pd.Timestamp("2022-10-01"), END)
}

#Hyperparameters
T_STEPS = 1000
BETA_START = 1e-4 #noise variance on first step
BETA_END = 0.02 #noise variance on last step
EPOCHS = 100  #how many times we iterate over the whole training set
BATCH_SIZE = 128  #Number of windows in training set
LR = 2e-4   #Learning rate for the optimizer
WIDTH = 64  #Number of hidden units in the MLP
COND_DIM = 128  #Dimension of the embedding of the conditioning variables (future features + region id)
COND_DROPOUT = 0.1  #Dropout rate for the conditioning variables, to make the model more robust to missing data
EMA_DECAY = 0.999   #Decay rate for the exponential moving average of the model parameters, used for evaluation
SEED = 0   #Random seed for reproducibility (like set seed in R)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

#For each region, we build the windows of past and future data that will be used for training, validation and testing
def build_windows(panel, period_start, period_end):
    X_past, C_fut, Y, RID = [], [], [], []
    for region, g in panel.groupby("region"):
        g = g.sort_values("date").reset_index(drop=True)
        past = g[PAST_FEATS].to_numpy(np.float32)
        fut  = g[FUT_FEATS].to_numpy(np.float32)
        targ = g[TARGETS].to_numpy(np.float32)
        rid  = int(g["region_id"].iloc[0])
        for t in range(INPUT_LEN, len(g) - OUTPUT_LEN + 1):
            tgt_start = g["date"].iloc[t]
            tgt_end   = g["date"].iloc[t + OUTPUT_LEN - 1]
            if tgt_start >= period_start and tgt_end <= period_end:
                X_past.append(past[t - INPUT_LEN:t])      # (60, 7)
                C_fut.append(fut[t:t + OUTPUT_LEN])        # (30, 4)
                Y.append(targ[t:t + OUTPUT_LEN])           # (30, 2)
                RID.append(rid)
    return (np.stack(X_past), np.stack(C_fut), np.stack(Y),
            np.array(RID, dtype=np.int64))

class WindowDataset(Dataset):
    def __init__(self, x_past, c_fut, y, rid):
        self.x_past = torch.from_numpy(x_past)
        self.c_fut = torch.from_numpy(c_fut)
        self.y = torch.from_numpy(y)
        self.rid = torch.tensor(rid)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_past[idx], self.c_fut[idx], self.y[idx], self.rid[idx]

#Noise schedule: linear from beta_start to beta_end
def schedule(t, b0, b1, device):
    betas = torch.linspace(b0, b1, t, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "abar": abar,
        "sqrt_abar": torch.sqrt(abar),
        "sqrt_one_minus_abar": torch.sqrt(1.0 - abar)
    }

#Model encoder
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

#Single vector for 60 days of past data + region id
class PastEncoder(nn.Module):
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.GELU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x_past):
        h = self.net(x_past.transpose(1, 2))   # (B, 64, 60)
        h = h.mean(dim=-1)                      # pooling over time -> (B, 64)
        return self.head(h)

#Condition injection: we embed the future features and region id into a vector of size cond_dim, that will be added to the past embedding
class ResBlock(nn.Module):
    def __init__(self, ch, cond_dim, fut_ch, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch + fut_ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * ch)

    def forward(self, h, cond, fut):
        scale, shift = self.film(cond)[..., None].chunk(2, dim=1)
        x = self.norm1(h)
        x = x * (1 + scale) + shift
        x = F.gelu(x)
        x = self.conv1(torch.cat([x, fut], dim=1))
        x = F.gelu(self.norm2(x))
        x = self.conv2(x)
        return h + x


class ConditionalDenoiser(nn.Module):
    def __init__(self, n_targets=2, n_fut=4, n_past=7,
                 width=WIDTH, cond_dim=COND_DIM, n_regions=N_REGIONS):
        super().__init__()
        self.cond_dim = cond_dim
        self.past_enc = PastEncoder(n_past, cond_dim)
        self.region_emb = nn.Embedding(n_regions, cond_dim)     #region id as categorical variable
        self.null_cond  = nn.Parameter(torch.zeros(cond_dim))
        self.t_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.fut_proj = nn.Conv1d(n_fut, width, 1)      # projection of future covariates
        self.in_conv  = nn.Conv1d(n_targets, width, 1)
        dilations = [1, 2, 4, 1, 2, 4]
        self.blocks = nn.ModuleList(
            [ResBlock(width, cond_dim, width, d) for d in dilations]
        )
        self.out = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv1d(width, n_targets, 1),
        )

    def forward(self, y_t, t, x_past, c_fut, rid, drop_mask=None):
        ctx = self.past_enc(x_past) + self.region_emb(rid)      # (B, cond_dim)
        fut = self.fut_proj(c_fut.transpose(1, 2))              # (B, width, 30)
        if drop_mask is not None:
            # Classifier-free guidance: for the dropped examples, we zero out the
            # conditions, so that the network learns to generate WITHOUT context.
            # At sampling, the two versions are combined to reinforce the effect.
            m = drop_mask.float()[:, None]
            ctx = torch.where(drop_mask[:, None], self.null_cond[None], ctx)
            fut = fut * (1 - m[..., None])
        cond = ctx + self.t_mlp(timestep_embedding(t, self.cond_dim))  # context + "how much noise"
        h = self.in_conv(y_t.transpose(1, 2))                  # (B, width, 30)
        for blk in self.blocks:
            h = blk(h, cond, fut)
        return self.out(h).transpose(1, 2)                     # predicted eps: (B, 30, 2)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
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
    B = y0.shape[0]
    # 1) choose a random timestep t for each example in the batch
    t = torch.randint(0, T_STEPS, (B,), device=DEVICE)
    # 2) sample true noise that we want the network to predict
    eps = torch.randn_like(y0)
    sa  = sched["sqrt_abar"][t][:, None, None]
    soma = sched["sqrt_one_minus_abar"][t][:, None, None]
    # 3) PROCESSO FORWARD: spoil the true target y0 up to level t in one go
    y_t = sa * y0 + soma * eps
    # 4) turn off the conditioning for a random subset of examples (classifier-free guidance)
    drop = (torch.rand(B, device=DEVICE) < COND_DROPOUT) if COND_DROPOUT > 0 else None
    eps_pred = model(y_t, t, x_past, c_fut, rid, drop_mask=drop)
    # 5) error = distance between true noise and predicted noise (the training metric)
    return F.mse_loss(eps_pred, eps)


def main():
    print(f"Device: {DEVICE}")
    panel = pd.read_csv(os.path.join(PROCESSED_DIR, "dataset_panel.csv"),
                        parse_dates=["date"])

    #Build the windows of past and future data for each split (train, val, test)
    data = {}
    for name, (s, e) in SPLITS.items():
        data[name] = build_windows(panel, s, e)
        print(f"{name}: {len(data[name][2])} finestre")

    train_ds = WindowDataset(*data["train"])
    val_ds   = WindowDataset(*data["val"])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    sched = schedule(T_STEPS, BETA_START, BETA_END, DEVICE)
    model = ConditionalDenoiser(n_targets=len(TARGETS),
                                n_fut=len(FUT_FEATS),
                                n_past=len(PAST_FEATS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ema = EMA(model, EMA_DECAY)

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # --- training ---
        model.train()
        tr = 0.0
        for batch in train_dl:
            opt.zero_grad()
            loss = diffusion_loss(model, sched, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # evita gradienti esplosivi
            opt.step()
            ema.update(model)                                        # aggiorna la copia EMA
            tr += loss.item() * batch[0].size(0)
        tr /= len(train_ds)

        # validation
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_dl:
                vl += diffusion_loss(model, sched, batch).item() * batch[0].size(0)
        vl /= len(val_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | train {tr:.4f} | val {vl:.4f}")

        # save the best model on validation (saving also the EMA weights)
        if vl < best_val:
            best_val = vl
            torch.save(
                {"model": model.state_dict(),
                 "ema": ema.shadow,
                 "config": {"T_STEPS": T_STEPS, "BETA_START": BETA_START,
                            "BETA_END": BETA_END, "WIDTH": WIDTH,
                            "COND_DIM": COND_DIM, "PAST_FEATS": PAST_FEATS,
                            "FUT_FEATS": FUT_FEATS, "TARGETS": TARGETS}},
                os.path.join(PROCESSED_DIR, "diffusion_ckpt.pt"))

    print(f"Fatto. Miglior val loss: {best_val:.4f}")
    print("Checkpoint salvato in processed_data/diffusion_ckpt.pt")


if __name__ == "__main__":
    main()