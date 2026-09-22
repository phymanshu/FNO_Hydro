#!/bin/bash
JOBID=$1
B=$2
VHLLE=/afs/cern.ch/work/h/hsharma/vhlle
EOS_OUT=/eos/user/h/hsharma/vhlle_output

kinit -R 2>/dev/null; aklog 2>/dev/null
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# use HTCondor sandbox (auto-created, local disk, fast)
# $TMPDIR or $PWD is already the sandbox
WORKDIR=$PWD
mkdir -p $WORKDIR/data

# copy vHLLE binary and EOS tables to local sandbox (read once)
cp $VHLLE/hlle_visc $WORKDIR/
cp -r $VHLLE/eos    $WORKDIR/

# write params
cat > $WORKDIR/params.dat << PEOF
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

# run entirely on local disk
cd $WORKDIR
./hlle_visc -params $WORKDIR/params.dat 2>&1 | tail -3

# copy only small files to EOS (ic2D=157K, skip freezeout=40MB)
mkdir -p $EOS_OUT/event_${JOBID}
cp $WORKDIR/data/ic2D.dat      $EOS_OUT/event_${JOBID}/ 2>/dev/null
cp $WORKDIR/data/freezeout.dat $EOS_OUT/event_${JOBID}/ 2>/dev/null
cp $WORKDIR/params.dat         $EOS_OUT/event_${JOBID}/ 2>/dev/null

echo "Done: event_${JOBID} b=${B}"
ls -lh $EOS_OUT/event_${JOBID}/
# sandbox cleaned up automatically by HTCondor
