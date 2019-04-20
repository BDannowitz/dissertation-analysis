#!/bin/bash

################
#define local variables based on where this script is
#note: trusts that script is in setup of seaquest-distribution repo
################
script_path="$(dirname "$(readlink -f ${BASH_SOURCE[0]})")"
nucdeproot="$(dirname $script_path/.)"
repopath="$(dirname $script_path/../../..)"
#distribroot="/your/path/to/seaquest-distribution/setup"
distribroot="$(dirname $repopath/seaquest-distribution/setup/.)"

if [ -d "$distribroot" ]; then
    source "$distribroot/setup.sh"
else
    echo "Please check that seaquest-distribution exists."
fi

export PYTHONPATH=`minidropit $PYTHONPATH $nucdeproot/pylib`
export PYTHONPATH=$nucdeproot/pylib:$PYTHONPATH
