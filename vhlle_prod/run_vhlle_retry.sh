#!/bin/bash
LINENO=$1
RETRY_IDS=/afs/cern.ch/work/h/hsharma/epos4/vhlle_prod/retry_ids.txt
VHLLE=/afs/cern.ch/work/h/hsharma/vhlle

read JOBID B <<< $(sed -n "$((LINENO+1))p" $RETRY_IDS)
OUTDIR=/afs/cern.ch/work/h/hsharma/epos4/vhlle_prod/output/event_${JOBID}

kinit -R 2>/dev/null; aklog 2>/dev/null
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# work entirely in /tmp - no AFS during vHLLE run
TMPDIR=$(mktemp -d /tmp/vhlle_${JOBID}_XXXXXX)
cd $TMPDIR

# symlink EOS tables
ln -sf $VHLLE/eos eos

# write params
cat > params.dat << PEOF
eosType      0
etaS         0.08
zetaS        0.0
e_crit       0.5
nx           61
ny           61
nz           21
xmin        -12.0
xmax         12.0
ymin        -12.0
ymax         12.0
etamin       -5.0
etamax        5.0
icModel      1
impactPar    $B
epsilon0     55.0
s0ScaleFactor 55.0
tau0         0.5
tauMax       15.0
dtau         0.2
PEOF

# pre-create data/ so ic2D.dat write succeeds
mkdir -p data

# run vHLLE - all output goes to /tmp
$VHLLE/hlle_visc -params $TMPDIR/params.dat 2>&1 | tail -5

# copy results to AFS
mkdir -p $OUTDIR
cp data/ic2D.dat      $OUTDIR/ 2>/dev/null && echo "ic2D copied" || echo "ic2D MISSING"
cp data/freezeout.dat $OUTDIR/ 2>/dev/null && echo "freezeout copied" || echo "freezeout MISSING"
cp params.dat         $OUTDIR/ 2>/dev/null

# cleanup /tmp
cd /tmp && rm -rf $TMPDIR
