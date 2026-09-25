#!/bin/bash
JOBID=$1
B=$2
SEED=$3
VHLLE=/afs/cern.ch/work/h/hsharma/vhlle
EOS_OUT=/eos/user/h/hsharma/vhlle_mcglauber/event_${JOBID}
MLDIR=/afs/cern.ch/work/h/hsharma/epos4/vhlle_prod

kinit -R 2>/dev/null; aklog 2>/dev/null
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

WORKDIR=$(mktemp -d /tmp/mcglauber_XXXXXX)
cd $WORKDIR
cp $VHLLE/hlle_visc .
cp -r $VHLLE/eos .
mkdir -p data

# generate MC Glauber IC
python3 $MLDIR/gen_mc_glauber_ic.py $B $SEED $WORKDIR/ic_input.dat

cat > params.dat << PEOF
eosType       0
etaS          0.08
zetaS         0.0
e_crit        0.5
nx            61
ny            61
nz            21
xmin         -12.0
xmax          12.0
ymin         -12.0
ymax          12.0
etamin        -5.0
etamax         5.0
icModel       2
glauberVar    0
s0ScaleFactor 1.0
tau0          0.5
tauMax        15.0
dtau          0.2
PEOF

./hlle_visc -params params.dat -ISinput $WORKDIR/ic_input.dat 2>&1 | tail -3

mkdir -p $EOS_OUT
cp data/ic2D.dat      $EOS_OUT/ 2>/dev/null
cp data/freezeout.dat $EOS_OUT/ 2>/dev/null
cp params.dat         $EOS_OUT/ 2>/dev/null
echo "$B" > $EOS_OUT/b_value.txt
echo "$SEED" > $EOS_OUT/seed.txt

cd /tmp && rm -rf $WORKDIR
