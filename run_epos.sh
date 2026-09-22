#!/bin/bash
JOBID=$1
WORKDIR=/afs/cern.ch/work/h/hsharma/epos4
OUTDIR=$WORKDIR/output

kinit -R 2>/dev/null; aklog 2>/dev/null
eval $(/cvmfs/alice.cern.ch/bin/alienv printenv EPOS4/v4.0.3-alice6-1)

TMPDIR=$(mktemp -d /tmp/epos4_${JOBID}_XXXXXX)
cd $TMPDIR

cp $WORKDIR/optns/PbPb_3050.optns .
sed -i "s/^set nfull.*/set nfull 1/" PbPb_3050.optns
echo "set ranseed $((JOBID * 12345 + 42))" >> PbPb_3050.optns

# copy EOS table - try both locations
cp $EPO4/src/KWt/eos1f.eos z-eos4f.eos || \
cp $EPO4/share/src/KWt/eos1f.eos z-eos4f.eos

# verify it exists before running
if [ ! -f z-eos4f.eos ]; then
    echo "ERROR: eos1f.eos not found at $(date)" >&2
    exit 1
fi

$EPO4/bin/epos -hepmc PbPb3050_job${JOBID} PbPb_3050

cp PbPb3050_job${JOBID}.hepmc $OUTDIR/ 2>/dev/null

cd /tmp && rm -rf $TMPDIR
