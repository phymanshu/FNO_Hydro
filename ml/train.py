"""
Phase 1 training: observable-level surrogate conditioned on b.
Input:  [pT histogram (27 bins), b (fm)]  → [B, 28]
Target: [N_ch, <pT>, v2, v3]             → [B, 4]
"""
import os, torch
import torch.nn as nn
from dataset import EPOS4Dataset, compute_observables
os.environ["OMP_NUM_THREADS"] = "1"

DATA_DIR  = "/afs/cern.ch/work/h/hsharma/epos4/output/"
SAVE_DIR  = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/"
N_EPOCHS  = 300
LR        = 1e-3
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

class ObsSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64),  nn.GELU(),
            nn.Linear(64, 4),
        )
    def forward(self, x): return self.net(x)

def event_to_obs(ev):
    px, py = ev[:, 0], ev[:, 1]
    pt  = torch.sqrt(px**2 + py**2)
    phi = torch.atan2(py, px)
    v2  = torch.sqrt(torch.cos(2*phi).mean()**2 + torch.sin(2*phi).mean()**2)
    v3  = torch.sqrt(torch.cos(3*phi).mean()**2 + torch.sin(3*phi).mean()**2)
    return torch.tensor([float(pt.shape[0]), pt.mean().item(),
                         v2.item(), v3.item()])

def event_to_hist(ev):
    pt = torch.sqrt(ev[:, 0]**2 + ev[:, 1]**2)
    h  = torch.histc(pt.clamp(0, 5), bins=27, min=0, max=5)
    return h / (h.sum() + 1e-8)

def train(centrality="30-50"):
    print(f"Device: {DEVICE}")
    print(f"Centrality: {centrality}%")
    ds = EPOS4Dataset(DATA_DIR, centrality=centrality)
    if len(ds) < 10:
        print("Need more events."); return

    # build input: [pT hist (27), b (1)] → 28 features
    X = torch.stack([
        torch.cat([event_to_hist(ev), torch.tensor([b])])
        for b, ev in ds.events
    ])
    Y = torch.stack([event_to_obs(ev) for b, ev in ds.events])

    # normalise
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8
    Xn = (X - X_mean) / X_std
    Yn = (Y - Y_mean) / Y_std

    print(f"\nEvents: {len(ds)}  Features: {X.shape[1]}")
    print(f"Target means:  N={Y_mean[0]:.0f}  pT={Y_mean[1]:.3f}  "
          f"v2={Y_mean[2]:.4f}  v3={Y_mean[3]:.4f}")

    # split
    idx     = torch.randperm(len(ds))
    n_val   = max(1, len(ds) // 5)
    n_train = len(ds) - n_val
    Xtr, Ytr = Xn[idx[:n_train]].to(DEVICE), Yn[idx[:n_train]].to(DEVICE)
    Xva, Yva = Xn[idx[n_train:]].to(DEVICE), Yn[idx[n_train:]].to(DEVICE)
    print(f"Train: {n_train}  Val: {n_val}")

    model   = ObsSurrogate().to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    loss_fn = nn.MSELoss()
    best    = float("inf")

    print(f"\nTraining {N_EPOCHS} epochs...")
    for ep in range(N_EPOCHS):
        model.train()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if (ep+1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xva), Yva).item()
            print(f"  Epoch {ep+1:3d}: train={loss.item():.4f}  "
                  f"val={vl:.4f}  lr={sched.get_last_lr()[0]:.2e}")
            if vl < best:
                best = vl
                torch.save({
                    "model": model.state_dict(),
                    "X_mean": X_mean, "X_std": X_std,
                    "Y_mean": Y_mean, "Y_std": Y_std,
                    "centrality": centrality,
                }, f"{SAVE_DIR}/best_obs_{centrality.replace('-','_')}.pt")
                print(f"    → saved")

    # final predictions
    model.eval()
    with torch.no_grad():
        pred_n = model(Xva).cpu()
    pred_real = pred_n * Y_std + Y_mean
    true_real = Yn[idx[n_train:]].cpu() * Y_std + Y_mean
    print(f"\nPredictions vs truth (val set):")
    for i, lab in enumerate(["N_ch", "<pT>", "v2", "v3"]):
        err = ((pred_real[:,i]-true_real[:,i]).abs()/true_real[:,i].abs()).mean()*100
        print(f"  {lab:6s}: pred={pred_real[:,i].mean():.4f}  "
              f"true={true_real[:,i].mean():.4f}  err={err:.1f}%")
    print(f"\nBest val loss: {best:.4f}")

if __name__ == "__main__":
    import sys
    cent = sys.argv[1] if len(sys.argv) > 1 else "30-50"
    train(cent)
