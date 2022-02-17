#!/bin/bash

set -ex

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR=$CDIR/..
PTDIR=${PTDIR:=$XDIR/..}

# unconditional patches
patch -d $PTDIR -p1 -i $XDIR/torch_patches/X10-codegen.diff -E -l -r - -s --no-backup-if-mismatch --follow-symlinks --force
