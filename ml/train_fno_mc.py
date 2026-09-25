import torch, numpy as np, sys, time, os
sys.path.insert(0, "/afs/cern.ch/work/h/hsharma/epos4/ml")
from fno import HydroFNO

# ── configuration ─────────────────────────────────────────────────────────
D_MODEL  = 32       # 32 (4.2M params) or 64 (16.8M params)
EPOCHS   = 200
BS       = 16       # training batch size
VAL_BS   = 8        # validation batch size
PATIENCE = 5        # early stopping: checks without improvement
LR       = 1e-3

CACHE = "/afs/cern.ch/work/h/hsharma/epos4/ml/cache_mcglauber.npz"
CKPT  = f"/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/fno_mcglauber_d{D_MODEL}.pt"
# ──────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# load cache
print("Loading cache...")
data=np.load(CACHE)
X_np=data['X']; Y_np=data['Y']
print(f"X={X_np.shape}  Y={Y_np.shape}")

# split BEFORE moving to GPU
N=len(X_np); N_val=500; N_train=N-N_val
rng=np.random.default_rng(42)
idx=rng.permutation(N)
train_idx=idx[:N_train]; val_idx=idx[N_train:]

X_train=torch.tensor(X_np[train_idx]).to(DEVICE)
Y_train=torch.tensor(Y_np[train_idx]).to(DEVICE)
X_val  =torch.tensor(X_np[val_idx]).to(DEVICE)
Y_val  =torch.tensor(Y_np[val_idx]).to(DEVICE)
print(f"Train: {N_train}  Val: {N_val}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# normalisation from TRAINING set only
X_mean=X_train.mean();  X_std=X_train.std()+1e-8
# per-channel normalisation for Y
Y_mean=Y_train.mean(dim=(0,2,3),keepdim=True)  # [1,4,1,1]
Y_std =Y_train.std(dim=(0,2,3),keepdim=True)+1e-8
print(f"X: mean={X_mean:.3f}  std={X_std:.3f}")
print(f"Y T-channel: mean={Y_mean[0,0,0,0]:.4f}  std={Y_std[0,0,0,0]:.4f}")

# model
model=HydroFNO(in_channels=1,out_channels=4,
               d_model=D_MODEL,n_blocks=4,
               modes1=16,modes2=16,n_cond=0).to(DEVICE)
n_params=sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params")

EPOCHS=200; BS=16; PATIENCE=5; VAL_BS=8
opt  =torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

best_val=1e9; no_improve=0; t0=time.time()
print(f"\nTraining {EPOCHS} epochs (BS={BS}, early stop patience={PATIENCE})")
print(f"{'Epoch':>6} {'Train':>9} {'Val':>9} {'Time':>7}")
print("-"*36)

for ep in range(EPOCHS):
    model.train()
    perm=torch.randperm(N_train,device=DEVICE)
    tl=[]
    for i in range(0,N_train,BS):
        bidx=perm[i:i+BS]
        X=X_train[bidx]; Y=Y_train[bidx]
        Xn=(X-X_mean)/X_std
        Yn=(Y-Y_mean)/Y_std
        pred=model(Xn)
        loss=torch.nn.functional.mse_loss(pred,Yn)
        opt.zero_grad(); loss.backward(); opt.step()
        tl.append(loss.item())
    sched.step()

    # validate in batches
    if (ep+1)%10==0 or ep==0:
        model.eval(); vl=[]
        with torch.no_grad():
            for vi in range(0,N_val,VAL_BS):
                xv=X_val[vi:vi+VAL_BS]; yv=Y_val[vi:vi+VAL_BS]
                pv=model((xv-X_mean)/X_std)
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
                "Y_mean":  Y_mean.cpu(), "Y_std":  Y_std.cpu(),
                "epoch":   ep+1, "val_loss": best_val,
                "d_model": 64,   "modes":    16,
                "ic_type": "MC_Glauber"
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
