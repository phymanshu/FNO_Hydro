# FNO_Hydro

**Fourier Neural Operator surrogate for viscous QGP hydrodynamics in Pb-Pb at LHC energies**

## Overview
FNO trained on vHLLE viscous hydro events to learn the mapping:
  ε(x,y, τ₀) → freeze-out surface T^μν(x,y, τ_f)

For Pb-Pb collisions at √s_NN = 5.36 TeV (LHC Run 3).

## Key Results
| Observable     | Value  |
|----------------|--------|
| T accuracy     | 1.5%   |
| ε₂ correlation | 0.993  |
| v2 correlation | 0.778  |
| GPU speedup    | ~200×  |
| Training time  | 17 min |

## Structure
- ml/          : FNO model, dataset parser, training scripts
- vhlle_prod/  : vHLLE production
- optns/       : EPOS4 configuration
- paper/       : paper draft

## Data
Large files (freezeout.dat, ic2D.dat, .hepmc) are on CERN EOS.
Model checkpoint available on request.

## Environment (CERN LXPLUS)
EPOS4:  source setup.sh
ML/GPU: source /cvmfs/sft.cern.ch/lcg/views/LCG_106a_cuda/x86_64-el9-gcc11-opt/setup.sh

## Reference
Extends Stewart & Putschke, PRC 113, 014904 (2026)
to viscous hydro at LHC energies.
