#!/usr/bin/env python
# Fully-integrated multi-modal survival pipeline
# – trains (or re-uses) modality encoders and then fine-tunes
#   a gated/concat fusion MLP-Cox model with warm-start logic.

from __future__ import annotations
import argparse, json, random, datetime, os, gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold, train_test_split
import optuna, shap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from optuna.trial import TrialState
import copy

# ───────────────── reproducibility / device ────────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_BN = False
PRELOADED_DATA = None
# ───────────────── constants / paths ───────────────────────────
EVENT_KEYS = ["AFE","ALE","MXE","SE","RI","A3SS","A5SS","HIT"]

DATA_ROOT = Path("/projectnb2/evolution/zwakefield/tcga/sir_analysis")
BASE_ROOT = DATA_ROOT / "survivalModel"
GEX_CSV   = DATA_ROOT / "cancerTypeHarmonized" / "GEX.csv"
ARNAP_CSV = {k: DATA_ROOT / "cancerTypeHarmonized" / f"{k}.csv" for k in EVENT_KEYS}
CLIN_CSV  = DATA_ROOT / "harmonized" / "clinical_harmonized_numeric.csv"

ENC_DIR   = DATA_ROOT / "survivalModel" / "encoders"
ENC_DIR.mkdir(parents=True, exist_ok=True)
ENC_OPTUNA = ENC_DIR / "optuna"
ENC_OPTUNA.mkdir(parents=True, exist_ok=True)

# ROOT_DIR  = DATA_ROOT / "survivalModel" / "ensemble"
# for p in ["models","optuna","logs","shap"]:
#     (ROOT_DIR / p).mkdir(parents=True, exist_ok=True)

PROJECT_CODE = {"ACC":1,"BLCA":2,"BRCA":3,"CESC":4,"CHOL":5,"COAD":6,"DLBC":7,"ESCA":8,
                "GBM":9,"HNSC":10,"KICH":11,"KIRC":12,"KIRP":13,"LAML":14,"LGG":15,
                "LIHC":16,"LUAD":17,"LUSC":18,"MESO":19,"OV":20,"PAAD":21,"PCPG":22,
                "PRAD":23,"READ":24,"SARC":25,"SKCM":26,"STAD":27,"TGCT":28,"THCA":29,
                "THYM":30,"UCEC":31,"UCS":32,"UVM":33}

def build_mod2idx(names:List[str])->Dict[str,torch.Tensor]:
    m2i=defaultdict(list)
    for j,n in enumerate(names):
        tag=n.split("::")[0]
        if tag not in EVENT_KEYS+["GEX"]: tag="CLIN"
        m2i[tag].append(j)
    return {k:torch.tensor(v,dtype=torch.long) for k,v in m2i.items()}

# ──────────────────── helpers ──────────────────────────────────
def minmax_01(x: torch.Tensor, eps: float = 1e-9):
    x_min = x.amin(dim=0, keepdim=True)
    x_rng = x.amax(dim=0, keepdim=True) - x_min + eps
    return (x - x_min) / x_rng, x_min, x_rng

def apply_minmax(x: torch.Tensor, x_min: torch.Tensor, x_rng: torch.Tensor):
    return (x - x_min) / x_rng

def scale_modalities(
    X: torch.Tensor,
    names: List[str],
    mod2idx: Dict[str, torch.Tensor],
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale selected modalities and return stats for reuse."""

    X = X.clone()
    x_min = torch.zeros(X.shape[1], dtype=X.dtype)
    x_rng = torch.ones(X.shape[1], dtype=X.dtype)

    # --- GEX: per-feature min–max -----------------------------------------
    g_idx = mod2idx.get("GEX")
    if g_idx is not None:
        g_min = X[:, g_idx].amin(0)
        g_rng = X[:, g_idx].amax(0) - g_min + eps
        X[:, g_idx] = (X[:, g_idx] - g_min) / g_rng
        x_min[g_idx] = g_min
        x_rng[g_idx] = g_rng

    # --- HIT: assume values in [-1, 1] -----------------------------------
    h_idx = mod2idx.get("HIT")
    if h_idx is not None:
        X[:, h_idx] = (X[:, h_idx] + 1.0) / 2.0
        x_min[h_idx] = -1.0
        x_rng[h_idx] = 2.0

    # --- AGE --------------------------------------------------------------
    try:
        age_idx = names.index("CLIN::AGE")
    except ValueError:
        age_idx = None
    if age_idx is not None:
        X[:, age_idx] = (X[:, age_idx] - 1.0) / (120.0 - 1.0)
        x_min[age_idx] = 1.0
        x_rng[age_idx] = 119.0

    return X, x_min, x_rng

def read_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")

def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    dummy = pd.get_dummies(series.fillna("MISSING"), prefix=prefix, drop_first=True)
    return dummy.reindex(sorted(dummy.columns), axis=1)

def strat_labels(t: np.ndarray, e: np.ndarray, n_q=4) -> np.ndarray:
    q = pd.qcut(t, n_q, labels=False, duplicates="drop")
    return q * 2 + e.astype(int)

def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor, eps=1e-8):
    risk  = risk - risk.max()
    order = torch.argsort(time, descending=True)
    pll = risk[order] - torch.logcumsumexp(risk[order], dim=0)
    return -(pll * event[order]).sum() / (event.sum() + eps)


def make_encoder(in_dim: int,
                 latent: int,
                 p_drop: float,
                 use_bn: bool = USE_BN) -> nn.Sequential:
    """
    Returns a 3-layer MLP (Linear → ReLU → [BatchNorm] → Dropout → Linear → ReLU).
    Setting use_bn=False gives you the BatchNorm-free version that worked
    during your early DAE experiments.
    """
    hid = max(512, in_dim // 4)
    layers = [nn.Linear(in_dim, hid), nn.ReLU()]
    if use_bn:
        layers.append(nn.BatchNorm1d(hid))
    layers += [nn.Dropout(p_drop), nn.Linear(hid, latent), nn.ReLU()]
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, in_dim: int, latent: int, p_drop: float):
        super().__init__()
        self.net = make_encoder(in_dim, latent, p_drop, use_bn=USE_BN)
    def forward(self, x):                 # unchanged
        return self.net(x)

# ───────────────── fusion heads ────────────────────────────────
class ConcatFusion(nn.Module):
    def __init__(self,dims:List[int],p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sum(dims),512), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(512,128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128,1,bias=False))
    def forward(self,zs): return self.net(torch.cat(zs,1)).squeeze(1)


class GateFusion(nn.Module):
    """
    Handles different latent widths by projecting every modality
    to a shared fusion dimension `fusion_lat`.
    """
    def __init__(self, dims: List[int], fusion_lat: int | None = None):
        super().__init__()

        # choose the fusion width
        self.fusion_lat = fusion_lat or max(dims)

        # (1) modality-specific projections -> common size
        self.proj = nn.ModuleList(
            [nn.Identity() if d == self.fusion_lat
             else nn.Linear(d, self.fusion_lat)
             for d in dims]
        )

        # (2) scalar gates
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.fusion_lat, 1), nn.Sigmoid())
             for _ in dims]
        )

        # (3) Cox head
        self.fin = nn.Linear(self.fusion_lat, 1, bias=False)

    def forward(self, zs):
        # zs[i] has shape (B, dims[i])  →  project → (B, fusion_lat)
        zs = [proj(z) for proj, z in zip(self.proj, zs)]
        gated = [gate(z) * z for gate, z in zip(self.gates, zs)]
        z_sum = torch.stack(gated, dim=0).sum(0)          # (B, fusion_lat)
        return self.fin(z_sum).squeeze(1)
# ───────────────── multi-modal wrapper ─────────────────────────
# class MultiModalCox(nn.Module):
#     def __init__(self,dims,cfg,fusion):
#         super().__init__()
#         lat,clin_lat,drop = cfg["latent"],cfg["clin_lat"],cfg["drop"]
#         self.enc_gex = Encoder(dims["GEX"],lat,drop)
#         self.enc_cln = Encoder(dims["CLIN"],clin_lat,drop)
#         self.enc_arp = nn.ModuleDict({k:Encoder(dims[k],lat,drop) for k in EVENT_KEYS})

#         latent_list = [lat] * len(EVENT_KEYS)
#         latent_list = [lat] + latent_list + [clin_lat]

#         self.fuse = ConcatFusion(latent_list,drop) if fusion=="concat" else GateFusion(latent_list,cfg.get("fusion_lat"))
#     def forward(self,batch):
#         zs = [self.enc_gex(batch["GEX"])] + [self.enc_arp[k](batch[k]) for k in EVENT_KEYS] + [self.enc_cln(batch["CLIN"])]
#         return self.fuse(zs)


class MultiModalCox(nn.Module):
    def __init__(self,dims,cfg,fusion):
        super().__init__()
        lat,clin_lat,drop = cfg["latent"],cfg["clin_lat"],cfg["drop"]
        self.enc_gex = Encoder(dims["GEX"],lat,drop)
        self.has_clin = "CLIN" in dims
        events = [k for k in EVENT_KEYS if dims.get(k,0) > 0]
        self.enc_cln = Encoder(dims["CLIN"],clin_lat,drop) if self.has_clin else None
        self.enc_arp = nn.ModuleDict({k:Encoder(dims[k],lat,drop) for k in events})
        self.events = events
        
        latent_list = [lat] + [lat]*len(events) + ([clin_lat] if self.has_clin else [])

        self.fuse = ConcatFusion(latent_list,drop) if fusion=="concat" else GateFusion(latent_list,cfg.get("fusion_lat"))
    def forward(self,batch):
        zs = [self.enc_gex(batch["GEX"])]
        zs += [self.enc_arp[k](batch[k]) for k in self.events]
        if self.has_clin:
            zs.append(self.enc_cln(batch["CLIN"]))
        return self.fuse(zs)



class _DAE(nn.Module):
    def __init__(self, in_dim: int, latent: int = 256, p_drop: float = 0.3):
        super().__init__()
        self.enc = make_encoder(in_dim, latent, p_drop, use_bn=False) 
        hid = self.enc[0].out_features                                # same hidden size
        self.dec = nn.Sequential(
            nn.Linear(latent, hid), nn.ReLU(),
            nn.Linear(hid, in_dim))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

class _VAE(nn.Module):
    def __init__(self,in_dim,latent=128):
        super().__init__()
        hid=max(512,in_dim//2)
        self.enc=nn.Sequential(nn.Linear(in_dim,hid),nn.ReLU())
        self.fc_mu, self.fc_log = nn.Linear(hid,latent), nn.Linear(hid,latent)
        self.dec=nn.Sequential(nn.Linear(latent,hid),nn.ReLU(),nn.Linear(hid,in_dim))
    def forward(self,x):
        h=self.enc(x); mu,logv=self.fc_mu(h),self.fc_log(h)
        z=mu+torch.exp(0.5*logv)*torch.randn_like(mu)
        recon=self.dec(z)
        kld=-0.5*(1+logv-mu.pow(2)-logv.exp()).mean()
        return recon,kld

def _train_autoencoder(
    loader,
    model,
    epochs,
    noise_p=0.1,
    *,
    lr=1e-3,
    weight_decay=0.0,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    is_vae = isinstance(model, _VAE)            # ⇠ only VAEs have a KL term

    model.train()
    for _ in range(epochs):
        for (x,) in loader:
            x = x.to(DEVICE)
            x_noisy = x + noise_p * torch.randn_like(x)

            if is_vae:
                recon, kld = model(x_noisy)     # two tensors
                loss = mse(recon, x) + kld
            else:                               # DAE
                recon = model(x_noisy)          # one tensor
                loss = mse(recon, x)

            opt.zero_grad()
            loss.backward()
            opt.step()


def apply_saved_scale(x, path):
    s = torch.load(path)
    return (x - s["min"].to(x.device)) / s["rng"].to(x.device)

def _save_encoder_weights(net: _DAE, path: Path) -> None:
    sd = { f"net.{k}": v for k, v in net.enc.state_dict().items() }  # add prefix once
    torch.save(sd, path)
def _ensure_encoders(
    enc_dir: Path,
    X: torch.Tensor,
    mod2idx: Dict[str, torch.Tensor],
    overwrite: bool,
    *,
    latent: int,
    clin_lat: int,
    drop: float = 0.3,
    lr: float = 1e-3,
    wd: float = 0.0,
    noise: float = 0.1,
    epochs: int = 70,
) -> None:
    """
    Train one DAE per modality on exactly the same MAD-filtered tensors that will
    feed the survival model, then save encoder weights into `enc_dir`.
    If the files already exist, nothing is done.
    """
    enc_dir.mkdir(parents=True, exist_ok=True)
    needed = ["GEX_encoder.pt", *[f"{k}_encoder.pt" for k in EVENT_KEYS]]

    if not overwrite and all((enc_dir / n).exists() for n in needed):
        print(f"[INFO] Re-using encoders in {enc_dir}")
        return

    if overwrite:
        print(f"[INFO] --overwrite: retraining encoders in {enc_dir}")
        # (optional) remove the old files first
        for n in needed:
            (enc_dir / n).unlink(missing_ok=True)

    print(f"[INFO] Pre-training encoders in {enc_dir} …")

    # ---------- helper that builds a loader directly from a Tensor ------------
    def _make_loader(x: torch.Tensor, bs: int = 512):
        ds = torch.utils.data.TensorDataset(x.cpu())
        return torch.utils.data.DataLoader(ds, batch_size=bs,
                                           shuffle=True, drop_last=True)

    # ---------- GEX -----------------------------------------------------------
    gex_block = X[:, mod2idx["GEX"]]
    dae = _DAE(gex_block.shape[1], latent=latent, p_drop=drop).to(DEVICE)
    _train_autoencoder(
        _make_loader(gex_block, bs=256),
        dae,
        epochs,
        noise_p=noise,
        lr=lr,
        weight_decay=wd,
    )
    _save_encoder_weights(dae, enc_dir / "GEX_encoder.pt")
    del dae; gc.collect(); torch.cuda.empty_cache()

    # ---------- each ARP block ------------------------------------------------
    for k in EVENT_KEYS:
        block = X[:, mod2idx[k]]
        if block.shape[1] == 0:
            continue                           # some cancers miss a splice type
        dae = _DAE(block.shape[1], latent=latent, p_drop=drop).to(DEVICE)
        _train_autoencoder(
            _make_loader(block, bs=256),
            dae,
            epochs,
            noise_p=noise,
            lr=lr,
            weight_decay=wd,
        )
        _save_encoder_weights(dae, enc_dir / f"{k}_encoder.pt")
        del dae; gc.collect(); torch.cuda.empty_cache()

    # clin_block = X[:, mod2idx["CLIN"]]
    # dae = _DAE(clin_block.shape[1], latent=clin_lat).to(DEVICE)
    # _train_autoencoder(_make_loader(clin_block, bs=512), dae, epochs=15)
    # _save_encoder_weights(dae, enc_dir / "CLIN_encoder.pt")
    # del dae; gc.collect(); torch.cuda.empty_cache()

    print("[INFO] Encoder pre-training complete")


def search_encoder_hparams(
    X: torch.Tensor,
    raw_idx: Dict[str, torch.Tensor],
    cancer: str,
    trials: int = 20,
) -> Dict[str, float]:
    """Optuna search over DAE pretraining hyperparameters.

    The following variables are tuned:
      - ``latent``  : encoder latent dimension
      - ``drop``    : dropout probability
      - ``lr``      : learning rate
      - ``wd``      : weight decay
      - ``noise``   : corruption noise for the DAE
      - ``epochs``  : number of pretraining epochs
      - ``mad_k``   : features kept per modality via MAD filtering
    """

    def mad_topk(block: torch.Tensor, k: int) -> torch.Tensor:
        med = block.median(0).values
        mad = (block - med).abs().median(0).values
        return torch.topk(mad, min(k, block.shape[1])).indices

    gex_raw = X[:, raw_idx["GEX"]]
    idx = np.arange(len(gex_raw))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        cfg = {
            "latent": trial.suggest_int("latent", 64, 512, step=64),
            "drop": trial.suggest_float("drop", 0.1, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "wd": trial.suggest_float("wd", 1e-6, 1e-3, log=True),
            "noise": trial.suggest_float("noise", 0.0, 0.2),
            "epochs": trial.suggest_int("epochs", 20, 100, step=20),
            "mad_k": trial.suggest_int("mad_k", 1000, 8000, step=1000),
        }

        # ----- evaluate DAE reconstruction on each modality separately -----
        mse = nn.MSELoss()
        total_loss = 0.0
        n_modalities = 0
        modalities = ["GEX", *EVENT_KEYS]

        for m in modalities:
            block = X[:, raw_idx[m]]
            if block.shape[1] == 0:
                continue

            feat_idx = mad_topk(block, cfg["mad_k"])
            tr = block[tr_idx][:, feat_idx]
            va = block[va_idx][:, feat_idx]

            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(tr),
                batch_size=256,
                shuffle=True,
                drop_last=True,
            )
            model = _DAE(tr.shape[1], latent=cfg["latent"], p_drop=cfg["drop"]).to(DEVICE)
            _train_autoencoder(
                loader,
                model,
                cfg["epochs"],
                noise_p=cfg["noise"],
                lr=cfg["lr"],
                weight_decay=cfg["wd"],
            )

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                va_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(va), batch_size=256
                )
                for (x,) in va_loader:
                    x = x.to(DEVICE)
                    recon = model(x)
                    va_loss += mse(recon, x).item() * len(x)

            total_loss += va_loss / len(va)
            n_modalities += 1

        return total_loss / max(n_modalities, 1)
    print(f"[INFO] Encoder db located: {ENC_OPTUNA}/{cancer}.db")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=f"sqlite:///{ENC_OPTUNA}/{cancer}.db",
        study_name=f"{cancer}_ENC",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=trials)

    print("[ENCODER SEARCH] best val loss:", study.best_value)
    print("[ENCODER SEARCH] params:", study.best_params)
    return study.best_params


def _add_prefix(state_dict: dict, prefix: str) -> dict:
    """Return a copy with `prefix` prepended to every key, unless it’s already there."""
    return { (k if k.startswith(prefix) else prefix + k): v
             for k, v in state_dict.items() }


def load_pretrained_encoders(net: MultiModalCox,
                             cancer: str,
                             enc_dir: Path) -> None:
    net.enc_gex.load_state_dict(torch.load(enc_dir/cancer/"GEX_encoder.pt"), strict=False)
    for k in EVENT_KEYS:
        pt = enc_dir/cancer/f"{k}_encoder.pt"
        if pt.exists():
            net.enc_arp[k].load_state_dict(torch.load(pt), strict=False)
    # net.enc_cln.load_state_dict(torch.load(enc_dir/cancer/"CLIN_encoder.pt"), strict=False)
# ───────────────── data load & mod-idx map ─────────────────────
def load_omics(code:int,with_clin:bool)->Tuple[torch.Tensor,np.ndarray,np.ndarray,List[str]]:
    clin=pd.read_csv(CLIN_CSV).set_index("File.ID")
    clin=clin[clin["Project.ID_code"]==code]
    clin=clin[clin["Sample.Type"]!="Solid Tissue Normal"]
    if clin.empty: raise RuntimeError(f"No samples for code {code}")

    mats,names=[],[]
    gex=read_matrix(GEX_CSV)[clin.index].T
    mats.append(gex); names+= [f"GEX::{c}" for c in gex.columns]
    for k in EVENT_KEYS:
        df=read_matrix(ARNAP_CSV[k])[clin.index].T
        mats.append(df); names+= [f"{k}::{c}" for c in df.columns]

    X=np.hstack(mats)

    if with_clin:
        frames,c_names=[],[]
        for col,pref in [("stage_code","STAGE"),("gender_code","GEND"),("race_code","RACE")]:
            oh = one_hot(clin[col], pref)
            frames.append(oh.values)
            c_names += list(oh.columns)

        # keep age unscaled for now – will apply formula later
        age = clin["age_at_diagnosis"].astype(np.float32)
        frames.append(age.values[:, None])
        c_names.append("AGE")

        X = np.hstack([X] + frames)
        names += [f"CLIN::{n}" for n in c_names]

    t=clin["OS.time"].to_numpy(np.float32)
    e=clin["OS.event"].to_numpy(np.float32)
    keep=(~np.isnan(X).any(1)) & ~np.isnan(t) & ~np.isnan(e)
    return torch.tensor(X[keep],dtype=torch.float32),t[keep],e[keep],names

def cv_objective_joint(trial, X, t, e, names, cancer, raw_idx, use_gate: bool):
    # ---- search space (tweak as needed) ----
    cfg = {
        "mad_k":     trial.suggest_int("mad_k", 1000, 8000, step=1000),
        "latent":    trial.suggest_int("latent", 64, 512, step=64),
        "clin_lat":  trial.suggest_int("clin_lat", 32, 128, step=16),
        "drop":      trial.suggest_float("drop", 0.1, 0.5),
        "lr":        trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd":        trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "epochs":    trial.suggest_int("epochs", 80, 320, step=40),
        # Optional: fusion_lat for GateFusion
        "fusion_lat": trial.suggest_int("fusion_lat", 64, 512, step=64) if use_gate else None,
        # Optional: number of hidden units/layers, etc.
    }

    # ----- build mod2idx with this trial's mad_k -----
    def mad_topk(block, k):
        med = block.median(0).values
        mad = (block - med).abs().median(0).values
        return torch.topk(mad, min(k, block.shape[1])).indices

    mod2idx = {
        "GEX":  raw_idx["GEX"][ mad_topk(X[:, raw_idx["GEX"]],  cfg["mad_k"]) ],
        "CLIN": raw_idx["CLIN"],
    }
    for k in EVENT_KEYS:
        mod2idx[k] = raw_idx[k][ mad_topk(X[:, raw_idx[k]], cfg["mad_k"]) ]

    y   = strat_labels(t, e)
    skf = StratifiedKFold(4, shuffle=True, random_state=SEED)
    fold_scores = []

    for tr_idx, va_idx in skf.split(np.arange(len(X)), y):
        # scale with training fold only
        X_tr_raw, X_va_raw = X[tr_idx], X[va_idx]
        X_tr, x_min, x_rng = scale_modalities(X_tr_raw.clone(), names, mod2idx)
        X_va               = apply_minmax(X_va_raw.clone(), x_min, x_rng)

        dims = {m: X[:, idx].shape[1] for m, idx in mod2idx.items()}
        fusion = "gate" if use_gate else "concat"
        net = MultiModalCox(dims, cfg, fusion).to(DEVICE)

        # no_pretrained ⇒ do NOT load weights; everything trains
        # (encoders already inside MultiModalCox)

        batch_tr = {m: X_tr[:, idx].to(DEVICE) for m, idx in mod2idx.items()}
        batch_va = {m: X_va[:, idx].to(DEVICE) for m, idx in mod2idx.items()}
        t_tr = torch.tensor(t[tr_idx]).to(DEVICE)
        e_tr = torch.tensor(e[tr_idx]).to(DEVICE)
        t_va = torch.tensor(t[va_idx]).to(DEVICE)
        e_va = torch.tensor(e[va_idx]).to(DEVICE)

        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

        best_c, stall = 0.0, 0
        for _ in range(cfg["epochs"]):
            net.train(); opt.zero_grad()
            loss = cox_ph_loss(net(batch_tr), t_tr, e_tr)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            net.eval()
            with torch.no_grad():
                risk_va = net(batch_va).cpu().numpy()
            c_idx = concordance_index(t_va.cpu(), -risk_va, e_va.cpu())

            if c_idx > best_c + 1e-4:
                best_c, stall = c_idx, 0
            else:
                stall += 1
            if stall >= 50:
                break

        fold_scores.append(best_c)
        del net; torch.cuda.empty_cache()

    return float(np.mean(fold_scores))

def cv_objective(trial, X, t, e, mod2idx, names, cancer, enc_cfg, no_pretrained, use_gate: bool):
    cfg = {
        "latent":  enc_cfg["latent"],
        "drop":    enc_cfg["drop"],
        "lr":      trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "wd":      trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "epochs":  trial.suggest_int("epochs", 80, 320, step=40),
        "clin_lat": trial.suggest_int("clin_lat", 32, 128, step=16),
        "warm": trial.suggest_int("warm", 0, 40, step=10),
        "enc_lr_scale": trial.suggest_float("enc_lr_scale", .05, 1.0, log=True),
        "fusion_lat": trial.suggest_int("fusion_lat", 64, 512, step=64) if use_gate else None,
    }

    y   = strat_labels(t, e)
    skf = StratifiedKFold(4, shuffle=True, random_state=SEED)
    fold_scores = []

    for tr_idx, va_idx in skf.split(np.arange(len(X)), y):
        # ----- NEW: fresh model each fold -----
        dims = {m: X[:, idx].shape[1] for m, idx in mod2idx.items()}
        
        fusion = "gate" if use_gate else "concat"
        net  = MultiModalCox(dims, cfg, fusion).to(DEVICE)
        
        if not no_pretrained:
            load_pretrained_encoders(net, cancer, ENC_DIR)

        head_params = list(net.fuse.parameters()) + list(net.enc_cln.parameters())
        enc_params  = list(net.enc_gex.parameters()) + list(net.enc_arp.parameters())

        warm = cfg['warm']

        if no_pretrained:
            # scratch: train encoders immediately
            for p in enc_params: p.requires_grad_(True)
        else:
            # pretrained: maybe freeze first
            if warm > 0:
                for p in enc_params: p.requires_grad_(False)

        base_lr = cfg["lr"]
        enc_lr  = base_lr if no_pretrained else base_lr * cfg.get("enc_lr_scale", 0.1)
        
        # ----- scale train + val identically -----
        X_tr, x_min, x_rng = scale_modalities(X[tr_idx].clone(), names, mod2idx)
        X_va               = apply_minmax(X[va_idx].clone(), x_min, x_rng)

        batch_tr = {m: X_tr[:, idx].to(DEVICE) for m, idx in mod2idx.items()}
        batch_va = {m: X_va[:, idx].to(DEVICE) for m, idx in mod2idx.items()}

        t_tr = torch.tensor(t[tr_idx]).to(DEVICE)
        e_tr = torch.tensor(e[tr_idx]).to(DEVICE)
        t_va = torch.tensor(t[va_idx]).to(DEVICE)
        e_va = torch.tensor(e[va_idx]).to(DEVICE)

        opt = torch.optim.AdamW([
            {"params": head_params, "lr": base_lr},
            {"params": enc_params,  "lr": enc_lr}
         ], weight_decay=cfg["wd"])
        
        best_c, stall = 0.0, 0
        for ep in range(cfg["epochs"]):

            if (not no_pretrained) and warm > 0 and ep == warm:
                for p in enc_params: 
                    p.requires_grad_(True)
                opt.param_groups[1]["lr"] = enc_lr
                
            net.train()
            opt.zero_grad()
            loss = cox_ph_loss(net(batch_tr), t_tr, e_tr)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            net.eval()
            with torch.no_grad():
                risk_va = net(batch_va).cpu().numpy()
            c_idx = concordance_index(t_va.cpu(), -risk_va, e_va.cpu())
            best_c = max(best_c, c_idx)
            stall  = 0 if c_idx == best_c else stall + 1
            if stall >= 50:
                break

        fold_scores.append(best_c)
        # ----- clean up -----
        del net; torch.cuda.empty_cache()

    return float(np.mean(fold_scores))
# ───────────────── fit_full with warm-start ─────────────────────
def fit_full(X_dev,t_dev,e_dev,mod2idx,cfg,model_dir,fusion,cancer,x_min,x_rng,no_pretrained):
    WARM=cfg.get("warm", 0)
    dims={m:X_dev[:,idx].shape[1] for m,idx in mod2idx.items()}
    net=MultiModalCox(dims,cfg,fusion).to(DEVICE)
    if not no_pretrained:
        load_pretrained_encoders(net,cancer, ENC_DIR)
    if USE_BN:
        for m in net.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()                             # stop running-stat updates
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

    tr_idx,va_idx=train_test_split(np.arange(len(X_dev)),test_size=0.25,
                                   stratify=strat_labels(t_dev,e_dev),random_state=SEED)

    batches={"tr":{m:X_dev[tr_idx][:,idx].to(DEVICE) for m,idx in mod2idx.items()},
             "va":{m:X_dev[va_idx][:,idx].to(DEVICE) for m,idx in mod2idx.items()}}
    t_tr,e_tr=torch.tensor(t_dev[tr_idx]).to(DEVICE),torch.tensor(e_dev[tr_idx]).to(DEVICE)
    t_va,e_va=torch.tensor(t_dev[va_idx]).to(DEVICE),torch.tensor(e_dev[va_idx]).to(DEVICE)

    head_params=list(net.fuse.parameters())+list(net.enc_cln.parameters())
    enc_params=list(net.enc_gex.parameters())+list(net.enc_arp.parameters())

    enc_scale_default = 0.1          # 10× smaller by default
    enc_scale = cfg.get("enc_lr_scale", enc_scale_default)
    
    if no_pretrained:
        enc_scale = cfg.get("enc_lr_scale_scratch", 1.0)
        # train encoders from scratch immediately
        for p in enc_params: p.requires_grad_(True)
        warm_cut = 0
    else:
        warm_cut = max(WARM, 0)
        if warm_cut > 0:
            for p in enc_params: p.requires_grad_(False)
                
    opt=torch.optim.AdamW([{"params":head_params,"lr":cfg["lr"]},
                           {"params":enc_params,"lr":cfg["lr"]*enc_scale}],
                           weight_decay=cfg["wd"])

    best_c = -1e9
    best_state = copy.deepcopy(net.state_dict())
    stall =0
    tloss,vloss,tcidx,vcidx=[],[],[],[]
    for ep in range(cfg["epochs"]):

        if (not no_pretrained) and ep == warm_cut and warm_cut > 0:
            for p in enc_params: 
                p.requires_grad_(True)
            opt.param_groups[1]["lr"] = cfg["lr"] * enc_scale

        net.train(); opt.zero_grad()
        risk_tr = net(batches["tr"])
        loss_tr=cox_ph_loss(risk_tr,t_tr,e_tr); loss_tr.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        opt.step()
        net.eval()
        with torch.no_grad():
            risk_va=net(batches["va"])
            loss_va=cox_ph_loss(risk_va,t_va,e_va)
            c_tr=concordance_index(t_dev[tr_idx],-net(batches["tr"]).cpu().numpy(),e_dev[tr_idx])
            c_va=concordance_index(t_dev[va_idx],-risk_va.cpu().numpy(),e_dev[va_idx])
        tloss.append(loss_tr.item()); vloss.append(loss_va.item())
        tcidx.append(c_tr); vcidx.append(c_va)
        if c_va>best_c: best_c,best_state,stall=c_va,copy.deepcopy(net.state_dict()),0
        else: stall+=1
        if stall>=50: break

    model_dir.mkdir(parents=True,exist_ok=True)
    plt.figure(); plt.plot(tloss,label="train"); plt.plot(vloss,label="val")
    plt.xlabel("epoch"); plt.ylabel("Cox-PH loss"); plt.legend()
    plt.tight_layout(); plt.savefig(model_dir/"loss_curve.png",dpi=150); plt.close()
    plt.figure(); plt.plot(tcidx,label="train"); plt.plot(vcidx,label="val"); plt.ylim(0.4,1.0)
    plt.xlabel("epoch"); plt.ylabel("C-index"); plt.legend()
    plt.tight_layout(); plt.savefig(model_dir/"cidx_curve.png",dpi=150); plt.close()

    net.load_state_dict(best_state)
    torch.save(net.state_dict(),model_dir/"best_model.pt")
    torch.save({"x_min":x_min.cpu(),"x_rng":x_rng.cpu(),"mod2idx":mod2idx},model_dir/"preproc.pt")
    return net,x_min,x_rng,best_c


def run_shap(
    net:           nn.Module,
    X:             torch.Tensor,
    mod2idx:       Dict[str, torch.Tensor],
    x_min:         torch.Tensor,
    x_rng:         torch.Tensor,
    feat_names:    List[str],
    out_dir:       Path,
    *,
    nsamples:      int  = 200,
    background:    int  = 400,
    risk_mode:     str  = "shift",   # "raw" | "shift" | "zscore"
    seed:          int  = 42,
) -> None:
    """
    SHAP for the multi-modal Cox net
    --------------------------------
    risk_mode:
        "raw"    – the untouched hazard logit  r(x)
        "shift"  – r(x) − μ   (μ = mean risk on background set)
        "zscore" – (r(x) − μ) / σ
    The **sign is flipped** ( −risk ) so *positive SHAP → higher hazard.*
    """

    rng = np.random.default_rng(seed)
    net.eval(); out_dir.mkdir(parents=True, exist_ok=True)

    # 0 ─── build a FLAT feature matrix in *fusion order* ------------------
    events = [k for k in EVENT_KEYS if k in mod2idx and len(mod2idx[k])>0]
    order = ["GEX", *events] + (["CLIN"] if "CLIN" in mod2idx else [])
    idx_lists = [mod2idx[m] for m in order if len(mod2idx[m]) > 0]
    concat_idx = torch.cat(idx_lists) if len(idx_lists) > 1 else idx_lists[0]
    
    X_raw = X[:, concat_idx]                             # for colour bar
    X_sc  = apply_minmax(X.clone(), x_min, x_rng)[:, concat_idx]

    # 1 ─── turn flat → dict blocks (so the model sees the right slices) ---
    slices, start = {}, 0
    for m in order:
        end = start + len(mod2idx[m])
        slices[m] = slice(start, end);  start = end

    class SliceWrap(nn.Module):
        def __init__(self, base, sl): super().__init__(); self.base, self.sl = base, sl
        def forward(self, x):
            xb = {m: x[:, s] for m, s in self.sl.items()}
            return self.base(xb)                         # shape (B,)

    wrapped = SliceWrap(net, slices).to(X.device)

    # ── sanity: gradients must be non-zero -------------------------------
    X_tmp = X_sc[:8].clone().requires_grad_(True).to(X.device)
    wrapped(X_tmp).sum().backward()
    g_max = X_tmp.grad.abs().max().item()
    if g_max == 0:
        print("⚠️  GRADIENTS ARE ZERO  – check slice indices!"); return
    print(f"✓ gradients flow; max|∇| = {g_max:.3e}")

    # 2 ─── choose background / foreground rows ---------------------------
    bg_idx = rng.choice(len(X_sc), size=min(background, len(X_sc)), replace=False)
    fg_idx = rng.choice(len(X_sc), size=min(nsamples, len(X_sc)), replace=False)

    X_back = X_sc[bg_idx].to(X.device)
    X_expl = X_sc[fg_idx].to(X.device)
    X_col  = X_raw[fg_idx]            # cpu later

    # 3 ─── optional centring / scaling of the risk -----------------------
    with torch.no_grad():
        risk_bg = wrapped(X_back)          # (B,)
        mu  = risk_bg.mean()
        sig = risk_bg.std().clamp_min(1e-9)

    class RiskHead(nn.Module):
        def __init__(self, base, mu, sig, mode):
            super().__init__(); self.base, self.mu, self.sig, self.mode = base, mu, sig, mode
        def forward(self, x):
            r = self.base(x)
            if self.mode == "shift":   r = r - self.mu
            elif self.mode == "zscore": r = (r - self.mu) / self.sig
            return (-r).unsqueeze(1)   # flip sign → ↑SHAP = ↑hazard

    head = RiskHead(wrapped, mu, sig, risk_mode).to(X.device)

    expl = shap.DeepExplainer(head, X_back)
    # expl = shap.GradientExplainer(head, X_back)
    sv   = expl.shap_values(X_expl, check_additivity=False)
    sv   = sv[0] if isinstance(sv, list) else sv      # (N,D)

    # 4 ─── diagnostics ----------------------------------------------------
    print("max |SHAP|:", float(np.abs(sv).max()),
          "   mean |SHAP|:", float(np.abs(sv).mean()))
    np.save(out_dir / "shap_values.npy", sv)

    # 5 ─── plot -----------------------------------------------------------
    shap.summary_plot(
        sv,
        X_col.cpu().numpy(),                    # colour uses *raw* values
        feature_names=[feat_names[i] for i in concat_idx],
        max_display=20,
        show=False
    )
    plt.savefig(out_dir / "summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(sv).mean(0)
    pd.Series(mean_abs, index=[feat_names[i] for i in concat_idx]) \
      .sort_values(ascending=False) \
      .to_csv(f"{out_dir}/shap_mean_abs.csv")
# ───────────────────────────────── main ─────────────────────────
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--cancer", required=True)
    pa.add_argument("--with_clin", action="store_true")
    pa.add_argument("--fusion", choices=["concat", "gate"], default="gate")
    pa.add_argument("--max_trials", type=int, default=0)
    pa.add_argument("--enc_trials", type=int, default=0)
    pa.add_argument("--use_preloaded", action="store_true")
    pa.add_argument("--overwrite", action="store_true")
    pa.add_argument("--shap_only", action="store_true")
    pa.add_argument("--no_pretrained", action="store_true")
    pa.add_argument("--exp_tag", default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    pa.add_argument("--root_dir", default=str(DATA_ROOT / "survivalModel" / "ensemble"))
    args = pa.parse_args()
    
    use_gate = (args.fusion == "gate")
    # ── paths ─────────────────────────────────────────────────────
    ROOT_DIR = Path(args.root_dir) / args.exp_tag
    for p in ["models", "optuna", "logs", "shap"]:
        (ROOT_DIR / p).mkdir(parents=True, exist_ok=True)

    cancer_enc_dir = ENC_DIR / args.cancer  # keep encoders global/reusable

    # ── load data ─────────────────────────────────────────────────
    code = PROJECT_CODE[args.cancer]
    if args.use_preloaded:
        assert PRELOADED_DATA is not None, "PRELOADED_DATA is empty!"
        X, t, e, names = PRELOADED_DATA
        print("[INFO] Using tensors from PRELOADED_DATA")
    else:
        X, t, e, names = load_omics(code, with_clin=args.with_clin)

    raw_idx = build_mod2idx(names)

    # helper
    def mad_topk_block(block: torch.Tensor, k: int, fit_rows: np.ndarray):
        med = block[fit_rows].median(0).values
        mad = (block[fit_rows] - med).abs().median(0).values
        return torch.topk(mad, min(k, block.shape[1])).indices

    # defaults (used in two-stage or fallback)
    enc_cfg = {
        "latent": 128,
        "drop":   0.3,
        "lr":     1e-3,
        "wd":     0.0,
        "noise":  0.1,
        "epochs": 70,
        "mad_k":  5000,
    }

    # ── SHAP only ─────────────────────────────────────────────────
    if args.shap_only:
        model_dir    = ROOT_DIR / "models" / args.cancer
        weights_path = model_dir / "best_model.pt"
        preproc_path = model_dir / "preproc.pt"
        if not (weights_path.exists() and preproc_path.exists()):
            raise FileNotFoundError("Need trained model for --shap_only")

        pre = torch.load(preproc_path, map_location="cpu")
        mod2idx = pre["mod2idx"]
        dims = {m: X[:, idx].shape[1] for m, idx in mod2idx.items()}
        dummy_cfg = {"latent": 64, "drop": 0.3, "clin_lat": 32}
        net = MultiModalCox(dims, dummy_cfg, args.fusion).to(DEVICE)
        net.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        net.eval()

        x_min, x_rng = pre["x_min"], pre["x_rng"]
        with torch.no_grad():
            r = net({m: X[:512, idx].to(DEVICE) for m, idx in mod2idx.items()})
        print("risk mean / std:", r.mean().item(), r.std().item())

        run_shap(net, X, mod2idx, x_min, x_rng, names, ROOT_DIR / "shap" / args.cancer)
        print("Done – SHAP artefacts written to", ROOT_DIR / "shap" / args.cancer)
        return

    # ── hold-out test split ───────────────────────────────────────
    tr_idx, te_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.15,
        stratify=strat_labels(t, e),
        random_state=SEED,
    )

    joint_search = args.no_pretrained and args.max_trials > 0

    # ──────────────────────────────────────────────────────────────
    # JOINT SEARCH (encoders + head tuned together)
    # ──────────────────────────────────────────────────────────────
    if joint_search:
        print("[OPTUNA] Joint search (encoders + head) because --no_pretrained")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            storage=f"sqlite:///{ROOT_DIR}/optuna/{args.cancer}_JOINT.db",
            study_name=f"{args.cancer}_JOINT",
            load_if_exists=True,
        )
        done = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
        todo = max(0, args.max_trials - done)
        print(f"[OPTUNA] {done}/{args.max_trials} done; running {todo} more")
        if todo > 0:
            study.optimize(
                lambda tr: cv_objective_joint(tr, X, t, e, names, args.cancer, raw_idx, use_gate),
                n_trials=todo
            )

        best = study.best_params
        print("Best C-index:", study.best_value)
        print("Params:", json.dumps(best, indent=2))

        enc_cfg = {k: best[k] for k in ["latent", "drop", "mad_k"]}
        cfg     = {k: best[k] for k in ["latent", "drop", "lr", "wd", "epochs", "clin_lat"]}
        if "fusion_lat" in best:
            cfg["fusion_lat"] = best["fusion_lat"]

        # recompute mod2idx on ALL dev rows with chosen mad_k
        fit_rows = tr_idx
        mod2idx = {
            "GEX":  raw_idx["GEX"][ mad_topk_block(X[:, raw_idx["GEX"]], enc_cfg["mad_k"], fit_rows) ],
            "CLIN": raw_idx["CLIN"],
        }
        for k in EVENT_KEYS:
            mod2idx[k] = raw_idx[k][ mad_topk_block(X[:, raw_idx[k]], enc_cfg["mad_k"], fit_rows) ]

        # scale dev/test
        X_dev_raw, X_test_raw = X[tr_idx], X[te_idx]
        X_dev, mi, rng = scale_modalities(X_dev_raw.clone(), names, mod2idx)
        X_test         = apply_minmax(X_test_raw.clone(), mi, rng)

    # ──────────────────────────────────────────────────────────────
    # TWO-STAGE PATH
    # ──────────────────────────────────────────────────────────────
    else:
        # encoder search / load
        if args.enc_trials > 0:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                storage=f"sqlite:///{ENC_OPTUNA}/{args.cancer}.db",
                study_name=f"{args.cancer}_ENC",
                load_if_exists=True,
            )
            done = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
            todo = max(0, args.enc_trials - done)
            if todo > 0:
                best = search_encoder_hparams(X, raw_idx, args.cancer, trials=todo)
                enc_cfg.update(best)
            else:
                enc_cfg.update(study.best_params)
        else:
            enc_db = ENC_OPTUNA / f"{args.cancer}.db"
            print(f"[INFO] Encoder dbs loaded from: {enc_db}")
            if enc_db.exists():
                try:
                    study = optuna.load_study(
                        study_name=f"{args.cancer}_ENC",
                        storage=f"sqlite:///{enc_db}",
                    )
                    enc_cfg.update(study.best_params)
                except Exception as ex:
                    print("[WARN] Could not load encoder study:", ex)

        # build mod2idx on train rows only
        fit_rows = tr_idx
        mod2idx = {
            "GEX":  raw_idx["GEX"][ mad_topk_block(X[:, raw_idx["GEX"]], enc_cfg["mad_k"], fit_rows) ],
            "CLIN": raw_idx["CLIN"],
        }
        for k in EVENT_KEYS:
            mod2idx[k] = raw_idx[k][ mad_topk_block(X[:, raw_idx[k]], enc_cfg["mad_k"], fit_rows) ]

        # split dev/test matrices
        X_dev_raw, X_test_raw = X[tr_idx], X[te_idx]
        t_dev, e_dev = t[tr_idx], e[tr_idx]
        X_dev, mi, rng = scale_modalities(X_dev_raw.clone(), names, mod2idx)
        X_test         = apply_minmax(X_test_raw.clone(), mi, rng)

        if args.max_trials != 0:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=SEED),
                storage=f"sqlite:///{ROOT_DIR}/optuna/{args.cancer}.db",
                study_name=f"{args.cancer}_MM",
                load_if_exists=True,
            )
            done = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
            todo = max(0, args.max_trials - done)
            print(f"[OPTUNA] {done}/{args.max_trials} trials finished; scheduling {todo} more")

            if todo > 0:
                study.optimize(
                    lambda tr: cv_objective(
                        tr, X_dev, t_dev, e_dev, mod2idx, names,
                        args.cancer, enc_cfg, args.no_pretrained, use_gate
                    ),
                    n_trials=todo
                )

            print("Best C-index:", study.best_value)
            print("Params:", json.dumps(study.best_params, indent=2))

            cfg = study.best_trial.params | {
                "latent": enc_cfg["latent"],
                "drop":   enc_cfg["drop"],
            }
        else:
            cfg = {
                "latent":  enc_cfg["latent"],
                "drop":    enc_cfg["drop"],
                "lr":      8e-5,
                "wd":      1e-4,
                "epochs":  200,
                "clin_lat":32,
            }

    print("[CONFIG]", json.dumps(cfg, indent=2))

    # ── Pretrain encoders only if using them ─────────────────────
    if not args.no_pretrained:
        _ensure_encoders(
            cancer_enc_dir, X_dev, mod2idx, args.overwrite,
            latent=enc_cfg["latent"], clin_lat=32,
            drop=enc_cfg["drop"], lr=enc_cfg["lr"],
            wd=enc_cfg["wd"], noise=enc_cfg["noise"],
            epochs=enc_cfg["epochs"],
        )

    # ── Full fit ─────────────────────────────────────────────────
    model_dir = ROOT_DIR / "models" / args.cancer
    net, x_min, x_rng, dev_c = fit_full(
        X_dev, t[tr_idx], e[tr_idx],
        mod2idx, cfg, model_dir,
        args.fusion, args.cancer, mi, rng,
        args.no_pretrained
    )

    # ── Test eval ────────────────────────────────────────────────
    batch_test = {m: X_test[:, idx].to(DEVICE) for m, idx in mod2idx.items()}
    net.eval()
    with torch.no_grad():
        risk = net(batch_test).cpu()
    c_test = concordance_index(t[te_idx], -risk.numpy(), e[te_idx])
    print(f"[RESULT] devC={dev_c:.3f}  testC={c_test:.3f}")

    # ── SHAP ─────────────────────────────────────────────────────
    run_shap(net, X, mod2idx, x_min, x_rng, names, ROOT_DIR / "shap" / args.cancer)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "cancer": args.cancer,
        "dev_C": round(float(dev_c), 4),
        "test_C": round(float(c_test), 4),
        "params": cfg,
        "enc_cfg": enc_cfg,
        "args": vars(args),
    }
    json.dump(meta, open(model_dir / "metrics.json", "w"), indent=2)
    print("Artefacts saved in", model_dir)

if __name__=="__main__":
    main()

