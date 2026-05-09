#!/bin/bash

set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# External repository / file initialization only.
# Run this once per fresh checkout before running CaMML experiments.

# download and setup causal-learn
if [ ! -d "causallearn" ]; then
  git clone https://github.com/py-why/causal-learn.git
  mv causal-learn/causallearn .
  rm -rf causal-learn
fi
cp scripts/ExactSearch.py causallearn/search/ScoreBased/ExactSearch.py

# download and setup minobsx
if [ ! -d "minobsx" ]; then
  git clone https://github.com/andrewli77/MINOBS-anc.git
  mv MINOBS-anc minobsx
fi
cp scripts/run-one-case-my.sh minobsx/run-one-case-my.sh
cp scripts/run-one-case-PG.sh minobsx/run-one-case-PG.sh
cd minobsx
mkdir -p anc_file out_BNs PGminobsx PGminobsx/parent_score PGminobsx/out_BNs PGminobsx/prior