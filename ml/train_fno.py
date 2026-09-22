"""
Phase 2 FNO training: ε(x,y,τ₀) → freeze-out T^μν field.
Uses vHLLE output: ic2D.dat (input) + freezeout.dat (target).

README: /eos/user/h/hsharma/epos4_plots/README_vhlle.md
"""
import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from fno import HydroFNO, PhysicsLoss
from vhlle_dataset import VHLLEDataset
os.environ["OMP_NUM_THREADS"] = "1"

PROD_DIR  = "/eos/user/h/hsharma/vhlle_output"
SAVE_DIR  = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/"
PLOT_DIR  = "/eos/user/h/hsharma/epos4_plots/"
N_EPOCHS  = 200
LR        = 1e-3
BATCH     = 8
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    print(f"Device: {DEVICE}")

    # load dataset
    ds = VHLLEDataset(PROD_DIR, verbose=True)
    if len(ds) < 10:
        print(f"Only {len(ds)} events — waiting for more vHLLE jobs to finish.")
        print(f"Check: condor_q 17391831")
        return

    # normalise X and Y
    all_X = torch.stack([ds[i][0] for i in range(len(ds))])
    all_Y = torch.stack([ds[i][1] for i in range(len(ds))])
    X_mean, X_std = all_X.mean(), all_X.std() + 1e-8
    Y_mean = all_Y.mean(dim=(0,2,3), keepdim=True)
    Y_std  = all_Y.std(dim=(0,2,3),  keepdim=True) + 1e-8

    print(f"\nDataset stats:")
    print(f"  X (ε): mean={X_mean:.3f}  std={X_std:.3f}  GeV/fm³")
    print(f"  Y (T) max channel mean: {Y_mean[0,0].mean():.4f} GeV")

    # normalised dataset
    class NormDataset(torch.utils.data.Dataset):
        def __init__(self, ds, X_mean, X_std, Y_mean, Y_std):
            self.ds = ds
            self.X_mean = X_mean; self.X_std = X_std
            self.Y_mean = Y_mean; self.Y_std = Y_std
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            X, Y = self.ds[i]
            return (X - self.X_mean) / self.X_std, (Y - self.Y_mean) / self.Y_std

    norm_ds = NormDataset(ds, X_mean, X_std, Y_mean, Y_std)

    # train/val split
    n_val   = max(1, len(ds) // 5)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(norm_ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)
    print(f"  Train: {n_train}  Val: {n_val}  Batches: {len(train_loader)}")

    # FNO model
    model = HydroFNO(
        in_channels  = 1,    # ε(x,y)
        out_channels = 4,    # T, ux, uy, cell_density
        d_model      = 32,   # smaller for CPU training
        n_blocks     = 4,
        modes1       = 10,
        modes2       = 10,
        n_cond       = 0,    # no conditioning for now
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FNO parameters: {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    best  = float("inf")

    print(f"\nTraining {N_EPOCHS} epochs...")
    for ep in range(N_EPOCHS):
        model.train()
        train_loss = 0
        for batch in train_loader:
            X, Y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            X = X.squeeze(1) if X.dim()==5 else X
            Y = Y.squeeze(1) if Y.dim()==5 else Y
            pred = model(X)
            loss = F.mse_loss(pred, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        sched.step()

        if (ep+1) % 20 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    X, Y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    X = X.squeeze(1) if X.dim()==5 else X
                    Y = Y.squeeze(1) if Y.dim()==5 else Y
                    val_loss += F.mse_loss(model(X), Y).item()
            val_loss /= len(val_loader)
            print(f"  Epoch {ep+1:3d}: train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")

            if val_loss < best:
                best = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "X_mean": X_mean, "X_std": X_std,
                    "Y_mean": Y_mean, "Y_std": Y_std,
                    "epoch": ep+1,
                }, f"{SAVE_DIR}/fno_vhlle_best.pt")
                print(f"    → saved (val={best:.4f})")

    print(f"\nBest val loss: {best:.4f}")
    print(f"Model: {SAVE_DIR}/fno_vhlle_best.pt")


if __name__ == "__main__":
    train()
