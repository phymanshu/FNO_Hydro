#!/bin/bash
# Usage:
#   source setup.sh        → EPOS4 mode (for running simulations)
#   source setup.sh ml     → ML mode (for Python/PyTorch)

kinit -R 2>/dev/null || echo "Note: run: kinit hsharma@CERN.CH"
aklog 2>/dev/null

export EPOS_WORK=/afs/cern.ch/work/h/hsharma/epos4
export EPOS_OUT=$EPOS_WORK/output
export EPOS_ML=$EPOS_WORK/ml
export OMP_NUM_THREADS=1

if [ "$1" == "ml" ]; then
    # ML mode: LCG only, no EPOS4
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
    TORCH_VER=$(cd /tmp && python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)
    echo "================================================"
    echo "  ML environment ready (PyTorch $TORCH_VER)"
    echo "  Output: $EPOS_OUT ($(ls $EPOS_OUT/*.hepmc 2>/dev/null | wc -l) hepmc files)"
    echo "================================================"
else
    # EPOS4 mode: alienv only, no LCG
    eval $(/cvmfs/alice.cern.ch/bin/alienv printenv EPOS4/v4.0.3-alice6-1 2>/dev/null)
    echo "================================================"
    echo "  EPOS4 environment ready"
    echo "  EPOS4: $EPO4"
    echo "  Output: $EPOS_OUT ($(ls $EPOS_OUT/*.hepmc 2>/dev/null | wc -l) hepmc files)"
    echo "================================================"
fi
