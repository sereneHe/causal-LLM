#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON_EXEC=${PYTHON_EXEC:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-/Users/xiaoyuhe/Causal-LLM/reports/outputs/KCRL/asia}
NB_EPOCH=${NB_EPOCH:-1}
INPUT_DIMENSION=${INPUT_DIMENSION:-64}
LAMBDA_ITER_NUM=${LAMBDA_ITER_NUM:-10}

PREP_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/data/preprocessing.py --dataset asia --raw-root /Users/xiaoyuhe/Causal-LLM/dataset/raw --processed-root /Users/xiaoyuhe/Causal-LLM/dataset/processed"
RUN_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/baseline/KCRL/kcrl_demo.py"

mkdir -p "${OUTPUT_DIR}"

${PREP_CMD}

export KCRL_DATA_PATH=/Users/xiaoyuhe/Causal-LLM/dataset/processed/asia
export KCRL_OUTPUT_DIR=${OUTPUT_DIR}
export KCRL_NB_EPOCH=${NB_EPOCH}
export KCRL_INPUT_DIMENSION=${INPUT_DIMENSION}
export KCRL_LAMBDA_ITER_NUM=${LAMBDA_ITER_NUM}
export KCRL_READ_DATA=1

${RUN_CMD}
