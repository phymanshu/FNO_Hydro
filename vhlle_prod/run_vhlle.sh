#!/bin/bash
JOBID=$1
B=$2
VHLLE=/afs/cern.ch/work/h/hsharma/vhlle
OUTDIR=/afs/cern.ch/work/h/hsharma/epos4/vhlle_prod/output/event_${JOBID}

kinit -R 2>/dev/null; aklog 2>/dev/null

# load LCG_106 for GSL libraries (required by vHLLE)
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

mkdir -p $OUTDIR

cat > $OUTDIR/params.dat << PEOF
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

cd $VHLLE
./hlle_visc -params $OUTDIR/params.dat -outputDir $OUTDIR 2>&1 | tail -3

rm -f $OUTDIR/outx.dat $OUTDIR/outy.dat $OUTDIR/outz.dat \
      $OUTDIR/outdiag.dat $OUTDIR/out.aniz.dat \
      $OUTDIR/outx.visc.dat $OUTDIR/outy.visc.dat \
      $OUTDIR/diag.visc.dat $OUTDIR/out2D.dat
