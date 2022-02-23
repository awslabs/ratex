#!/bin/bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ROOTPATH=$SCRIPTPATH/../../

python3 -m pylint $ROOTPATH/torch_mnm --rcfile=$ROOTPATH/scripts/lint/pylintrc \
    && python3 -m pylint $ROOTPATH/tests/python --rcfile=$ROOTPATH/scripts/lint/pytestlintrc
