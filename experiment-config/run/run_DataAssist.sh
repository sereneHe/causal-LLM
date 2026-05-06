#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON_EXEC=${PYTHON_EXEC:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-/Users/xiaoyuhe/Causal-LLM/reports/outputs/DataAssist/SmBFO}

PREP_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/data/preprocessing.py --dataset SmBFO --raw-root /Users/xiaoyuhe/Causal-LLM/dataset/raw --processed-root /Users/xiaoyuhe/Causal-LLM/dataset/processed"
RUN_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/utils/metrics.py --data-npy /Users/xiaoyuhe/Causal-LLM/dataset/processed/SmBFO/data.npy --output-dir ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

${PREP_CMD}
${RUN_CMD}
