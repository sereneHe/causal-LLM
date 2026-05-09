#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export KMP_DUPLICATE_LIB_OK=TRUE

CMD="uv run python $REPO_ROOT/experiment-config/run_experiment.py --multirun"
LLM_MINOBSX_PROBLEMS=${LLM_MINOBSX_PROBLEMS:-"asia,child,insurance,alarm,cancer,mildew,water,barley"}

cd "$REPO_ROOT"
${CMD} experiment='LLM_MINOBSx_experiment' problem="${LLM_MINOBSX_PROBLEMS}" problem.generator='LLM_MINOBSx'
