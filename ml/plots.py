"""
Comprehensive physics + ML diagnostic plots for EPOS4 surrogate project.
Run: python3 plots.py
Outputs: /afs/cern.ch/work/h/hsharma/epos4/ml/plots/
"""
import os, sys, math, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from dataset import EPOS4Dataset, observables

os.environ["OMP_NUM_THREADS"] = "1"
DATA_DIR  = "/afs/cern.ch/work/h/hsharma/epos4/output/"
PLOT_DIR  = "/eos/user/h/hsharma/epos4_plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

ALICE_BLUE = "#0072B2"
EPOS_RED   = "#D55E00"
FS         = 14
plt.rcParams.update({"font.size": FS, "axes.labelsize": FS,
                     "axes.titlesize": FS+1, "legend.fontsize": FS-2,
                     "figure.dpi": 120})

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = EPOS4Dataset(DATA_DIR, verbose=False)
events = ds.events
N_ev   = len(events)
print(f"  {N_ev} events loaded")

# ── Compute per-event observables ─────────────────────────────────────────────
print("Computing observables...")
Nch, mean_pT, v2s, v3s, v4s = [], [], [], [], []
eta_all, pt_all, phi_all, pid_all = [], [], [], []
species_count = Counter()

PDG_NAMES = {
    211: "π⁺", -211: "π⁻", 111: "π⁰",
    321: "K⁺", -321: "K⁻", 130: "K⁰L", 310: "K⁰S",
    2212: "p",  -2212: "p̄", 2112: "n", -2112: "n̄",
    3122: "Λ",  -3122: "Λ̄", 22: "γ",
}
IDX_TO_PDG = {}
from dataset import PDG_TO_IDX
for pdg, idx in PDG_TO_IDX.items():
    IDX_TO_PDG[int(idx)] = pdg

for ev in events:
    px, py, pz, E = ev[:,0], ev[:,1], ev[:,2], ev[:,3]
    pt  = torch.sqrt(px**2 + py**2)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + 1e-8))
    pid_idx = ev[:,4].long()

    # flow
    q2x = torch.cos(2*phi).mean(); q2y = torch.sin(2*phi).mean()
    q3x = torch.cos(3*phi).mean(); q3y = torch.sin(3*phi).mean()
    q4x = torch.cos(4*phi).mean(); q4y = torch.sin(4*phi).mean()

    Nch.append(pt.shape[0])
    mean_pT.append(pt.mean().item())
    v2s.append(math.sqrt(q2x**2 + q2y**2))
    v3s.append(math.sqrt(q3x**2 + q3y**2))
    v4s.append(math.sqrt(q4x**2 + q4y**2))

    # sample particles for global distributions (max 2000 per event)
    n_sample = min(2000, len(pt))
    idx_s    = torch.randperm(len(pt))[:n_sample]
    pt_all.extend(pt[idx_s].tolist())
    eta_all.extend(eta[idx_s].tolist())
    phi_all.extend(phi[idx_s].tolist())
    pid_all.extend(pid_idx[idx_s].tolist())

    for p in pid_idx.tolist():
        pdg = IDX_TO_PDG.get(p, 0)
        species_count[PDG_NAMES.get(pdg, str(pdg))] += 1

Nch     = np.array(Nch)
mean_pT = np.array(mean_pT)
v2s     = np.array(v2s)
v3s     = np.array(v3s)
v4s     = np.array(v4s)
pt_all  = np.array(pt_all)
eta_all = np.array(eta_all)
phi_all = np.array(phi_all)

print(f"  Done. {len(pt_all):,} particles sampled for distributions")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Single-particle distributions (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 1: Single-particle distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(f"EPOS4 Pb-Pb 30-50% √s_NN=5.36 TeV  ({N_ev} events)", fontweight="bold")

# pT spectrum
ax = axes[0,0]
pt_cut = pt_all[pt_all < 5]
ax.hist(pt_cut, bins=80, range=(0,5), color=EPOS_RED, alpha=0.8, density=True)
ax.set_xlabel("p_T (GeV/c)"); ax.set_ylabel("dN/dp_T (normalized)")
ax.set_title("Transverse momentum spectrum")
ax.set_yscale("log"); ax.set_xlim(0, 5)
ax.axvline(np.mean(pt_cut), color="black", ls="--", lw=1.5,
           label=f"⟨p_T⟩ = {np.mean(pt_cut):.3f} GeV")
ax.legend()

# η distribution
ax = axes[0,1]
eta_cut = eta_all[np.abs(eta_all) < 6]
ax.hist(eta_cut, bins=80, range=(-6,6), color=ALICE_BLUE, alpha=0.8, density=True)
ax.set_xlabel("η (pseudorapidity)"); ax.set_ylabel("dN/dη (normalized)")
ax.set_title("Pseudorapidity distribution")
ax.set_xlim(-6, 6)
ax.axvline(0, color="black", ls=":", lw=1)

# φ distribution
ax = axes[1,0]
ax.hist(phi_all, bins=72, range=(-np.pi, np.pi), color="green", alpha=0.8, density=True)
ax.set_xlabel("φ (azimuthal angle)"); ax.set_ylabel("dN/dφ (normalized)")
ax.set_title("Azimuthal angle distribution")
ax.set_xlim(-np.pi, np.pi)
ax.axhline(1/(2*np.pi), color="black", ls="--", lw=1.5, label="Uniform")
ax.legend()

# pT vs η 2D
ax = axes[1,1]
mask = (np.abs(eta_all) < 5) & (pt_all < 4)
h = ax.hist2d(eta_all[mask], pt_all[mask], bins=[60,50],
              range=[[-5,5],[0,4]], cmap="hot", norm=LogNorm())
plt.colorbar(h[3], ax=ax, label="counts")
ax.set_xlabel("η"); ax.set_ylabel("p_T (GeV/c)")
ax.set_title("p_T vs η correlation")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_single_particle.png", bbox_inches="tight")
plt.close()
print(f"  Saved 01_single_particle.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: Event-level observables
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 2: Event-level observables...")
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(f"EPOS4 Pb-Pb 30-50% — Event Observables ({N_ev} events)", fontweight="bold")

# Multiplicity
ax = axes[0,0]
ax.hist(Nch, bins=40, color=EPOS_RED, alpha=0.8, edgecolor="darkred", lw=0.5)
ax.set_xlabel("N_ch (charged particles)"); ax.set_ylabel("Events")
ax.set_title("Charged particle multiplicity")
ax.axvline(Nch.mean(), color="black", ls="--", lw=2,
           label=f"⟨N_ch⟩ = {Nch.mean():.0f}")
ax.legend()

# Mean pT
ax = axes[0,1]
ax.hist(mean_pT, bins=40, color=ALICE_BLUE, alpha=0.8, edgecolor="darkblue", lw=0.5)
ax.set_xlabel("⟨p_T⟩ per event (GeV/c)"); ax.set_ylabel("Events")
ax.set_title("Mean transverse momentum")
ax.axvline(mean_pT.mean(), color="black", ls="--", lw=2,
           label=f"mean = {mean_pT.mean():.3f} GeV")
ax.legend()

# v2
ax = axes[0,2]
ax.hist(v2s, bins=40, color="purple", alpha=0.8, edgecolor="indigo", lw=0.5)
ax.set_xlabel("v₂ (elliptic flow)"); ax.set_ylabel("Events")
ax.set_title("Elliptic flow coefficient v₂")
ax.axvline(v2s.mean(), color="black", ls="--", lw=2,
           label=f"⟨v₂⟩ = {v2s.mean():.4f}")
ax.legend()

# v3
ax = axes[1,0]
ax.hist(v3s, bins=40, color="darkorange", alpha=0.8, edgecolor="saddlebrown", lw=0.5)
ax.set_xlabel("v₃ (triangular flow)"); ax.set_ylabel("Events")
ax.set_title("Triangular flow coefficient v₃")
ax.axvline(v3s.mean(), color="black", ls="--", lw=2,
           label=f"⟨v₃⟩ = {v3s.mean():.4f}")
ax.legend()

# v2 vs Nch
ax = axes[1,1]
ax.scatter(Nch, v2s, alpha=0.4, s=15, color=EPOS_RED)
ax.set_xlabel("N_ch"); ax.set_ylabel("v₂")
ax.set_title("v₂ vs multiplicity")
z = np.polyfit(Nch, v2s, 1)
xline = np.linspace(Nch.min(), Nch.max(), 100)
ax.plot(xline, np.polyval(z, xline), "k--", lw=2,
        label=f"slope={z[0]:.2e}")
ax.legend()

# mean pT vs Nch
ax = axes[1,2]
ax.scatter(Nch, mean_pT, alpha=0.4, s=15, color=ALICE_BLUE)
ax.set_xlabel("N_ch"); ax.set_ylabel("⟨p_T⟩ (GeV/c)")
ax.set_title("⟨p_T⟩ vs multiplicity")
z = np.polyfit(Nch, mean_pT, 1)
ax.plot(xline, np.polyval(z, xline), "k--", lw=2)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_event_observables.png", bbox_inches="tight")
plt.close()
print(f"  Saved 02_event_observables.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: Flow coefficients
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 3: Flow coefficients...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("EPOS4 Pb-Pb 30-50% — Anisotropic Flow", fontweight="bold")

# v2 vs v3 scatter
ax = axes[0]
sc = ax.scatter(v2s, v3s, c=Nch, cmap="viridis", alpha=0.6, s=20)
plt.colorbar(sc, ax=ax, label="N_ch")
ax.set_xlabel("v₂"); ax.set_ylabel("v₃")
ax.set_title("v₂ vs v₃ correlation")

# vn distribution comparison
ax = axes[1]
bins = np.linspace(0, 0.15, 40)
ax.hist(v2s, bins=bins, alpha=0.7, color="purple",     label="v₂", density=True)
ax.hist(v3s, bins=bins, alpha=0.7, color="darkorange", label="v₃", density=True)
ax.hist(v4s, bins=bins, alpha=0.7, color="green",      label="v₄", density=True)
ax.set_xlabel("v_n"); ax.set_ylabel("Probability density")
ax.set_title("Flow harmonics v₂, v₃, v₄")
ax.legend()

# Azimuthal anisotropy: φ distribution with v2 modulation
ax = axes[2]
phi_bins = np.linspace(-np.pi, np.pi, 37)
phi_centers = 0.5*(phi_bins[:-1] + phi_bins[1:])
phi_hist, _ = np.histogram(phi_all, bins=phi_bins, density=True)
ax.plot(phi_centers, phi_hist, "o-", color=EPOS_RED, ms=5, label="EPOS4")
mean_v2 = v2s.mean()
phi_theory = (1 + 2*mean_v2*np.cos(2*phi_centers)) / (2*np.pi)
ax.plot(phi_centers, phi_theory, "k--", lw=2,
        label=f"1+2⟨v₂⟩cos(2φ), ⟨v₂⟩={mean_v2:.3f}")
ax.set_xlabel("φ"); ax.set_ylabel("dN/dφ")
ax.set_title("Azimuthal modulation")
ax.legend(); ax.set_xlim(-np.pi, np.pi)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_flow.png", bbox_inches="tight")
plt.close()
print(f"  Saved 03_flow.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Particle species
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 4: Particle species...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EPOS4 Pb-Pb 30-50% — Particle Species", fontweight="bold")

# species bar chart
ax = axes[0]
top_species = species_count.most_common(12)
names  = [s[0] for s in top_species]
counts = [s[1] for s in top_species]
total  = sum(counts)
colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
bars = ax.bar(names, [c/total*100 for c in counts], color=colors, edgecolor="black", lw=0.5)
ax.set_ylabel("Fraction (%)")
ax.set_title("Particle species composition")
ax.tick_params(axis="x", rotation=45)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{count/total*100:.1f}%", ha="center", va="bottom", fontsize=9)

# pT spectra by species
ax = axes[1]
species_pt = {name: [] for name, _ in top_species[:6]}
for ev in events:
    px, py = ev[:,0], ev[:,1]
    pt = torch.sqrt(px**2 + py**2)
    pid_idx = ev[:,4].long()
    for i, pt_val in enumerate(pt.tolist()):
        pdg  = IDX_TO_PDG.get(pid_idx[i].item(), 0)
        name = PDG_NAMES.get(pdg, "")
        if name in species_pt:
            species_pt[name].append(pt_val)

pt_bins = np.linspace(0, 3, 31)
pt_centers = 0.5*(pt_bins[:-1] + pt_bins[1:])
colors6 = plt.cm.tab10(np.linspace(0, 1, 6))
for (name, _), col in zip(top_species[:6], colors6):
    pts = np.array(species_pt[name])
    if len(pts) < 10: continue
    h, _ = np.histogram(pts, bins=pt_bins, density=True)
    ax.plot(pt_centers, h+1e-6, "-o", ms=3, lw=1.5, color=col, label=name)
ax.set_xlabel("p_T (GeV/c)"); ax.set_ylabel("dN/dp_T (normalized)")
ax.set_title("p_T spectra by species")
ax.set_yscale("log"); ax.set_xlim(0, 3); ax.legend(ncol=2)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_species.png", bbox_inches="tight")
plt.close()
print(f"  Saved 04_species.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Two-particle correlations (ΔηΔφ ridge)
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 5: Two-particle correlations (ΔηΔφ)...")
deta_all, dphi_all = [], []
n_pairs_per_event = 500

for ev in events[:min(100, N_ev)]:
    px, py, pz = ev[:,0], ev[:,1], ev[:,2]
    pt  = torch.sqrt(px**2 + py**2)
    phi = torch.atan2(py, px)
    eta = torch.asinh(pz / (pt + 1e-8))

    # select charged hadrons with pT > 0.2 GeV, |η| < 2.4
    mask = (pt > 0.2) & (eta.abs() < 2.4)
    phi_sel = phi[mask]; eta_sel = eta[mask]
    if len(phi_sel) < 10: continue

    n = min(100, len(phi_sel))
    idx1 = torch.randperm(len(phi_sel))[:n]
    idx2 = torch.randperm(len(phi_sel))[:n]

    for i, j in zip(idx1[:n_pairs_per_event].tolist(),
                    idx2[:n_pairs_per_event].tolist()):
        if i == j: continue
        dphi = phi_sel[i].item() - phi_sel[j].item()
        dphi = (dphi + np.pi) % (2*np.pi) - np.pi
        deta = eta_sel[i].item() - eta_sel[j].item()
        deta_all.append(deta); dphi_all.append(dphi)

deta_all = np.array(deta_all)
dphi_all = np.array(dphi_all)

fig = plt.figure(figsize=(13, 5))
fig.suptitle("EPOS4 Pb-Pb 30-50% — Two-particle Correlations", fontweight="bold")

ax1 = fig.add_subplot(121, projection="3d")
H, xedges, yedges = np.histogram2d(deta_all, dphi_all, bins=[40,36],
                                    range=[[-4,4],[-np.pi,np.pi]])
xc = 0.5*(xedges[:-1]+xedges[1:])
yc = 0.5*(yedges[:-1]+yedges[1:])
X, Y = np.meshgrid(xc, yc)
ax1.plot_surface(X.T, Y.T, H/H.mean(), cmap="jet", alpha=0.9)
ax1.set_xlabel("Δη"); ax1.set_ylabel("Δφ"); ax1.set_zlabel("C(Δη,Δφ)")
ax1.set_title("ΔηΔφ correlation (3D)")

ax2 = fig.add_subplot(122)
im = ax2.pcolormesh(yc, xc, H/H.mean(), cmap="jet", shading="auto")
plt.colorbar(im, ax=ax2, label="C(Δη,Δφ)")
ax2.set_xlabel("Δφ"); ax2.set_ylabel("Δη")
ax2.set_title("ΔηΔφ correlation (2D)")
ax2.axvline(0,  color="white", ls="--", lw=1, alpha=0.5)
ax2.axvline(np.pi, color="white", ls="--", lw=1, alpha=0.5)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_two_particle_corr.png", bbox_inches="tight")
plt.close()
print(f"  Saved 05_two_particle_corr.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: ML training diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 6: ML diagnostics...")
ckpt_path = "/afs/cern.ch/work/h/hsharma/epos4/ml/checkpoints/best_obs.pt"

if os.path.exists(ckpt_path):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    Y_mean = ckpt["Y_mean"]; Y_std = ckpt["Y_std"]

    # recompute predictions on all data
    from train import ObsSurrogate, event_to_hist, event_to_obs
    model = ObsSurrogate()
    model.load_state_dict(ckpt["model"])
    model.eval()

    X_mean = ckpt["X_mean"]; X_std = ckpt["X_std"]
    X_all  = torch.stack([event_to_hist(e) for e in events])
    Y_all  = torch.stack([event_to_obs(e)  for e in events])
    Xn     = (X_all - X_mean) / X_std

    with torch.no_grad():
        pred_n = model(Xn)
    pred = pred_n * Y_std + Y_mean
    true = Y_all

    labels = ["N_ch", "⟨p_T⟩ (GeV)", "v₂", "v₃"]
    units  = ["", "GeV", "", ""]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("ML Surrogate — Prediction vs Truth", fontweight="bold")

    for i, (lab, unit) in enumerate(zip(labels, units)):
        # pred vs true scatter
        ax = axes[0, i]
        ax.scatter(true[:,i].numpy(), pred[:,i].numpy(),
                   alpha=0.4, s=15, color=EPOS_RED)
        lims = [min(true[:,i].min(), pred[:,i].min()).item(),
                max(true[:,i].max(), pred[:,i].max()).item()]
        ax.plot(lims, lims, "k--", lw=2, label="ideal")
        err = ((pred[:,i]-true[:,i]).abs()/true[:,i].abs()).mean()*100
        ax.set_xlabel(f"True {lab}"); ax.set_ylabel(f"Predicted {lab}")
        ax.set_title(f"{lab}  (err={err:.1f}%)")
        ax.legend()

        # residual distribution
        ax = axes[1, i]
        residuals = ((pred[:,i]-true[:,i])/true[:,i]).numpy() * 100
        ax.hist(residuals, bins=30, color=ALICE_BLUE, alpha=0.8, edgecolor="navy")
        ax.axvline(0, color="black", ls="--", lw=2)
        ax.axvline(residuals.mean(), color="red", ls="-", lw=2,
                   label=f"mean={residuals.mean():.1f}%")
        ax.set_xlabel(f"Relative residual (%)"); ax.set_ylabel("Events")
        ax.set_title(f"{lab} residuals")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/06_ml_predictions.png", bbox_inches="tight")
    plt.close()
    print(f"  Saved 06_ml_predictions.png")
else:
    print(f"  No checkpoint found, skipping ML plot")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Observable correlations matrix
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 7: Correlation matrix...")
import itertools

obs_matrix = np.column_stack([Nch, mean_pT, v2s, v3s, v4s])
obs_labels = ["N_ch", "⟨p_T⟩", "v₂", "v₃", "v₄"]
n_obs = len(obs_labels)

fig, axes = plt.subplots(n_obs, n_obs, figsize=(13, 12))
fig.suptitle("EPOS4 Observable Correlation Matrix", fontweight="bold")

for i, j in itertools.product(range(n_obs), range(n_obs)):
    ax = axes[i, j]
    if i == j:
        ax.hist(obs_matrix[:,i], bins=25, color=EPOS_RED, alpha=0.8)
        ax.set_title(obs_labels[i], fontsize=10)
    elif i > j:
        ax.scatter(obs_matrix[:,j], obs_matrix[:,i],
                   alpha=0.3, s=8, color=ALICE_BLUE)
        corr = np.corrcoef(obs_matrix[:,j], obs_matrix[:,i])[0,1]
        ax.text(0.05, 0.92, f"r={corr:.2f}", transform=ax.transAxes,
                fontsize=9, color="red")
    else:
        corr = np.corrcoef(obs_matrix[:,j], obs_matrix[:,i])[0,1]
        ax.text(0.5, 0.5, f"{corr:.3f}", transform=ax.transAxes,
                ha="center", va="center", fontsize=14,
                color="red" if abs(corr) > 0.3 else "black",
                fontweight="bold" if abs(corr) > 0.3 else "normal")
        ax.set_facecolor(plt.cm.RdBu(0.5 + corr/2))
    if i == n_obs-1: ax.set_xlabel(obs_labels[j], fontsize=9)
    if j == 0:       ax.set_ylabel(obs_labels[i], fontsize=9)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_correlation_matrix.png", bbox_inches="tight")
plt.close()
print(f"  Saved 07_correlation_matrix.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Event-by-event fluctuations
# ═══════════════════════════════════════════════════════════════════════════════
print("Plot 8: Event-by-event fluctuations...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("EPOS4 Pb-Pb 30-50% — Event-by-Event Fluctuations", fontweight="bold")

# N_ch fluctuations
ax = axes[0]
ax.hist(Nch, bins=35, density=True, color=EPOS_RED, alpha=0.8, label="EPOS4")
mu, sig = Nch.mean(), Nch.std()
x = np.linspace(Nch.min(), Nch.max(), 200)
ax.plot(x, np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)),
        "k--", lw=2, label=f"Gaussian μ={mu:.0f} σ={sig:.0f}")
ax.set_xlabel("N_ch"); ax.set_ylabel("P(N_ch)")
ax.set_title(f"Multiplicity fluctuations\nσ/μ = {sig/mu:.3f}")
ax.legend()

# v2 fluctuations (Bessel-Gaussian shape expected)
ax = axes[1]
ax.hist(v2s, bins=35, density=True, color="purple", alpha=0.8, label="EPOS4")
ax.set_xlabel("v₂"); ax.set_ylabel("P(v₂)")
ax.set_title(f"v₂ fluctuations\n⟨v₂⟩={v2s.mean():.4f}  σ(v₂)={v2s.std():.4f}")
ax.legend()

# v2/v3 ratio
ax = axes[2]
ratio = v2s / (v3s + 1e-6)
ratio_cut = ratio[ratio < 10]
ax.hist(ratio_cut, bins=35, density=True, color="darkorange", alpha=0.8)
ax.set_xlabel("v₂/v₃"); ax.set_ylabel("P(v₂/v₃)")
ax.set_title(f"v₂/v₃ ratio\nmean = {ratio_cut.mean():.2f}")
ax.axvline(ratio_cut.mean(), color="black", ls="--", lw=2)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_fluctuations.png", bbox_inches="tight")
plt.close()
print(f"  Saved 08_fluctuations.png")

print(f"\n{'='*50}")
print(f"All plots saved to: {PLOT_DIR}")
print(f"Files: {sorted(os.listdir(PLOT_DIR))}")
