#!/bin/bash
# Condor wrapper: run the transferred stochaxtic package from the
# job's scratch directory.  Usage: run_omega.sh <arguments of python -m stochaxtic>
set -euo pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python -m stochaxtic "$@"
