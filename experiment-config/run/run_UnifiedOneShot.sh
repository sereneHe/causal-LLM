#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON_EXEC=${PYTHON_EXEC:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-/Users/xiaoyuhe/Causal-LLM/reports/outputs/UnifiedOneShot/asia}
ENABLED_MODELS=${ENABLED_MODELS:-PC}
TMP_DIR=${TMP_DIR:-${OUTPUT_DIR}/tmp_inputs}
METHOD_DIR="/Users/xiaoyuhe/Causal-LLM/scripts/baseline/Causal-LLM_Unified_One-Shot_Framework/Dag_generation and model_evaluation"

PREP_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/data/preprocessing.py --dataset asia --raw-root /Users/xiaoyuhe/Causal-LLM/dataset/raw --processed-root /Users/xiaoyuhe/Causal-LLM/dataset/processed"
RUN_CMD="cd \"${METHOD_DIR}\" && ${PYTHON_EXEC} main_runner.py --ground-truth-path ${TMP_DIR}/adj.csv --gaussian-path ${TMP_DIR}/samples.csv --output-dir ${OUTPUT_DIR} --enabled-models ${ENABLED_MODELS}"

mkdir -p "${OUTPUT_DIR}"
${PREP_CMD}
mkdir -p "${TMP_DIR}"
${PYTHON_EXEC} - <<PY
from pathlib import Path
import numpy as np
import pandas as pd
processed = Path("/Users/xiaoyuhe/Causal-LLM/dataset/processed/asia")
tmp = Path("${TMP_DIR}")
x = np.load(processed / "X.npy", allow_pickle=True)
y = np.load(processed / "adj.npy", allow_pickle=True)
pd.DataFrame(x).to_csv(tmp / "samples.csv", index=False)
pd.DataFrame(y).to_csv(tmp / "adj.csv", index=False, header=False)
PY
eval "${RUN_CMD}"
