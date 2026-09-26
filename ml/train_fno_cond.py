"""
Conditional FNO training: learns eta/s-dependent freeze-out prediction.
Caches dataset to RAM for fast training.
"""
import torch, numpy as np, sys, time, os
sys.path.insert(0, "/afs/cern.ch/work/h/hsharma/epos4/ml")
from fno_cond import HydroFNOCond
from vhlle_dataset_cond import VHLLEDatasetCond
import warnings; warnings.filterwarnings("ignore")

# ── configuration ──────────────────────────────────────────────────────────
D_MODEL  = 32
EPOCHS   = 300
BS       = 16
VAL_BS   = 8
PATIENCE = 8
LR       = 1e-3
CKPT = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/fno_cond.pt"
# ───────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# load and cache dataset
print("\nLoading dataset...")
ds = VHLLEDatasetCond(verbose=True)
if len(ds) < 10:
    print("Not enough data yet — wait for HTCondor jobs to finish")
    sys.exit(0)

# cache to RAM
print("Caching to RAM...")
X_list=[]; E_list=[]; Y_list=[]
for i in range(len(ds)):
    X,e,Y=ds[i]
    X_list.append(X.numpy())
    E_list.append(e.numpy())
    Y_list.append(Y.numpy())
    if (i+1)%500==0: print(f"  {i+1}/{len(ds)}", flush=True)

X_np=np.stack(X_list).astype(np.float32)
E_np=np.stack(E_list).astype(np.float32)
Y_np=np.stack(Y_list).astype(np.float32)
print(f"Cached: X={X_np.shape} E={E_np.shape} Y={Y_np.shape}")

# print eta/s distribution
etas_unique, counts = np.unique(E_np[:,0], return_counts=True)
print("eta/s distribution:")
for e,c in zip(etas_unique, counts):
    print(f"  eta/s={e:.3f}: {c} events")

# split
N=len(X_np); N_val=min(500, N//5)
rng=np.random.default_rng(42)
idx=rng.permutation(N)
train_idx=idx[N_val:]; val_idx=idx[:N_val]

# move to GPU
X_train=torch.tensor(X_np[train_idx]).to(DEVICE)
E_train=torch.tensor(E_np[train_idx]).to(DEVICE)
Y_train=torch.tensor(Y_np[train_idx]).to(DEVICE)
X_val  =torch.tensor(X_np[val_idx]).to(DEVICE)
E_val  =torch.tensor(E_np[val_idx]).to(DEVICE)
Y_val  =torch.tensor(Y_np[val_idx]).to(DEVICE)
N_train=len(train_idx)
print(f"\nTrain: {N_train}  Val: {N_val}")

# normalisation
X_mean=X_train.mean(); X_std=X_train.std()+1e-8
E_mean=E_train.mean(); E_std=E_train.std()+1e-8
Y_mean=Y_train.mean(dim=(0,2,3),keepdim=True)
Y_std =Y_train.std(dim=(0,2,3),keepdim=True)+1e-8
print(f"X: mean={X_mean:.3f}  std={X_std:.3f}")
print(f"E: mean={E_mean:.4f}  std={E_std:.4f}")

# model
model=HydroFNOCond(in_channels=1,out_channels=4,
                   d_model=D_MODEL,n_blocks=4,
                   modes1=16,modes2=16,cond_dim=1).to(DEVICE)
n_params=sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M parameters")

opt  =torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

best_val=1e9; no_improve=0; t0=time.time()
print(f"\nTraining {EPOCHS} epochs (BS={BS}, patience={PATIENCE})...")
print(f"{'Epoch':>6} {'Train':>9} {'Val':>9} {'Time':>7}")
print("-"*36)

for ep in range(EPOCHS):
    model.train()
    perm=torch.randperm(N_train,device=DEVICE)
    tl=[]
    for i in range(0,N_train,BS):
        bidx=perm[i:i+BS]
        X=X_train[bidx]; E=E_train[bidx]; Y=Y_train[bidx]
        Xn=(X-X_mean)/X_std
        En=(E-E_mean)/E_std
        Yn=(Y-Y_mean)/Y_std
        pred=model(Xn,En)
        loss=torch.nn.functional.mse_loss(pred,Yn)
        opt.zero_grad(); loss.backward(); opt.step()
        tl.append(loss.item())
    sched.step()

    if (ep+1)%10==0 or ep==0:
        torch.cuda.empty_cache()
        model.eval(); vl=[]
        with torch.no_grad():
            for vi in range(0,N_val,VAL_BS):
                xv=X_val[vi:vi+VAL_BS]
                ev=E_val[vi:vi+VAL_BS]
                yv=Y_val[vi:vi+VAL_BS]
                pv=model((xv-X_mean)/X_std,(ev-E_mean)/E_std)
                vl.append(torch.nn.functional.mse_loss(
                    pv,(yv-Y_mean)/Y_std).item())
        vl_mean=float(np.mean(vl))
        tl_mean=float(np.mean(tl))
        dt=time.time()-t0
        marker=""
        if vl_mean<best_val:
            best_val=vl_mean; no_improve=0
            torch.save({
                "model":   model.state_dict(),
                "X_mean":  X_mean.cpu(), "X_std":  X_std.cpu(),
                "E_mean":  E_mean.cpu(), "E_std":  E_std.cpu(),
                "Y_mean":  Y_mean.cpu(), "Y_std":  Y_std.cpu(),
                "epoch":   ep+1, "val_loss": best_val,
                "d_model": D_MODEL, "etas_trained": list(etas_unique),
            }, CKPT)
            marker=" ✓"
        else:
            no_improve+=1
        print(f"{ep+1:>6} {tl_mean:>9.4f} {vl_mean:>9.4f} {dt:>6.0f}s{marker}",
              flush=True)
        if no_improve>=PATIENCE:
            print(f"Early stopping at epoch {ep+1}")
            break

print(f"\nBest val loss: {best_val:.4f}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print(f"Saved: {CKPT}")
print(f"Trained on eta/s: {list(etas_unique)}")
