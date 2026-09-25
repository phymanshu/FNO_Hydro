"""
Stage 2 PointNet training: predict ux, uy at raw freeze-out cells.
Uses FNO T field as context. Caches dataset to RAM for fast training.
"""
import torch, numpy as np, sys, time, os
sys.path.insert(0, "/afs/cern.ch/work/h/hsharma/epos4/ml")
from pointnet import PointNetStage2
from pointnet_dataset import PointNetDataset
import warnings; warnings.filterwarnings("ignore")

# ── config ─────────────────────────────────────────────────────────────────
N_EVENTS  = 2000   # events to use (balance speed vs accuracy)
N_PTS     = 4096   # points per event
EPOCHS    = 200
BS        = 32
VAL_BS    = 16
PATIENCE  = 8
LR        = 1e-3
CKPT = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/pointnet_stage2.pt"
CACHE= "/afs/cern.ch/work/h/hsharma/epos4/ml/cache_pointnet.npz"
# ───────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# build or load cache
if os.path.exists(CACHE):
    print(f"Loading cache: {CACHE}")
    d = np.load(CACHE)
    pts_np=d['pts']; ux_np=d['ux']; uy_np=d['uy']
else:
    print(f"Building cache ({N_EVENTS} events × {N_PTS} pts)...")
    ds = PointNetDataset(max_events=N_EVENTS, n_pts=N_PTS, verbose=True)
    pts_list=[]; ux_list=[]; uy_list=[]
    for i in range(len(ds)):
        pts,ux,uy,_ = ds[i]
        if pts.abs().sum()==0: continue
        pts_list.append(pts.numpy())
        ux_list.append(ux.numpy())
        uy_list.append(uy.numpy())
        if (i+1)%200==0: print(f"  {i+1}/{len(ds)}", flush=True)
    pts_np = np.stack(pts_list).astype(np.float32)
    ux_np  = np.stack(ux_list).astype(np.float32)
    uy_np  = np.stack(uy_list).astype(np.float32)
    np.savez_compressed(CACHE, pts=pts_np, ux=ux_np, uy=uy_np)
    print(f"Saved cache: {CACHE}  ({pts_np.nbytes/1e6:.0f} MB)")

print(f"Dataset: {len(pts_np)} events  shape={pts_np.shape}")

# split
N=len(pts_np); N_val=min(300,N//5)
rng=np.random.default_rng(42); idx=rng.permutation(N)
tr=idx[N_val:]; va=idx[:N_val]

# move to GPU
pts_tr=torch.tensor(pts_np[tr]).to(DEVICE)
ux_tr =torch.tensor(ux_np[tr]).to(DEVICE)
uy_tr =torch.tensor(uy_np[tr]).to(DEVICE)
pts_va=torch.tensor(pts_np[va]).to(DEVICE)
ux_va =torch.tensor(ux_np[va]).to(DEVICE)
uy_va =torch.tensor(uy_np[va]).to(DEVICE)
N_train=len(tr)
print(f"Train: {N_train}  Val: {N_val}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# normalisation of targets
ux_mean=ux_tr.mean(); ux_std=ux_tr.std()+1e-8
uy_mean=uy_tr.mean(); uy_std=uy_tr.std()+1e-8
print(f"ux: mean={ux_mean:.4f}  std={ux_std:.4f}")
print(f"uy: mean={uy_mean:.4f}  std={uy_std:.4f}")

# model
model=PointNetStage2(in_dim=3, hidden=128, out_dim=2).to(DEVICE)
n=sum(p.numel() for p in model.parameters())
print(f"Model: {n/1e6:.3f}M parameters")

opt  =torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)

best_val=1e9; no_improve=0; t0=time.time()
print(f"\nTraining {EPOCHS} epochs (BS={BS}, patience={PATIENCE})...")
print(f"{'Epoch':>6} {'Train':>9} {'Val':>9} {'ux_MAE':>8} {'uy_MAE':>8} {'Time':>7}")
print("-"*52)

for ep in range(EPOCHS):
    model.train()
    perm=torch.randperm(N_train,device=DEVICE)
    tl=[]
    for i in range(0,N_train,BS):
        bidx=perm[i:i+BS]
        pts=pts_tr[bidx]
        ux_t=(ux_tr[bidx]-ux_mean)/ux_std
        uy_t=(uy_tr[bidx]-uy_mean)/uy_std
        pred=model(pts)  # [B, N_pts, 2]
        loss=(torch.nn.functional.mse_loss(pred[:,:,0],ux_t)+
              torch.nn.functional.mse_loss(pred[:,:,1],uy_t))
        opt.zero_grad(); loss.backward(); opt.step()
        tl.append(loss.item())
    sched.step()

    if (ep+1)%10==0 or ep==0:
        torch.cuda.empty_cache()
        model.eval(); vl=[]; ux_maes=[]; uy_maes=[]
        with torch.no_grad():
            for vi in range(0,N_val,VAL_BS):
                pts_v=pts_va[vi:vi+VAL_BS]
                ux_v =ux_va[vi:vi+VAL_BS]
                uy_v =uy_va[vi:vi+VAL_BS]
                pred_v=model(pts_v)
                ux_pred=pred_v[:,:,0]*ux_std+ux_mean
                uy_pred=pred_v[:,:,1]*uy_std+uy_mean
                vl.append((
                    torch.nn.functional.mse_loss(
                        pred_v[:,:,0],(ux_v-ux_mean)/ux_std)+
                    torch.nn.functional.mse_loss(
                        pred_v[:,:,1],(uy_v-uy_mean)/uy_std)
                ).item())
                ux_maes.append((ux_pred-ux_v).abs().mean().item())
                uy_maes.append((uy_pred-uy_v).abs().mean().item())

        vl_m=float(np.mean(vl))
        ux_mae=float(np.mean(ux_maes))
        uy_mae=float(np.mean(uy_maes))
        tl_m=float(np.mean(tl))
        dt=time.time()-t0
        marker=""
        if vl_m<best_val:
            best_val=vl_m; no_improve=0
            torch.save({
                "model":    model.state_dict(),
                "ux_mean":  ux_mean.cpu(), "ux_std": ux_std.cpu(),
                "uy_mean":  uy_mean.cpu(), "uy_std": uy_std.cpu(),
                "epoch":    ep+1, "val_loss": best_val,
            }, CKPT)
            marker=" ✓"
        else:
            no_improve+=1
        print(f"{ep+1:>6} {tl_m:>9.4f} {vl_m:>9.4f} "
              f"{ux_mae:>8.4f} {uy_mae:>8.4f} {dt:>6.0f}s{marker}",
              flush=True)
        if no_improve>=PATIENCE:
            print(f"Early stopping at epoch {ep+1}")
            break

print(f"\nBest val loss: {best_val:.4f}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
print(f"Saved: {CKPT}")
print(f"\nFor comparison — FNO ux/uy MAE:")
print(f"  ux: 0.158  (15%)")
print(f"  uy: 0.108  (11%)")
print(f"PointNet target: < 0.07 for both")
