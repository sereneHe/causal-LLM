#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

PYTHON_EXEC=${PYTHON_EXEC:-python3}
OUTPUT_DIR=${OUTPUT_DIR:-/Users/xiaoyuhe/Causal-LLM/reports/outputs/GUIDE/gaussian_30}
NUM_EPOCHS=${NUM_EPOCHS:-1}
HIDDEN_DIM=${HIDDEN_DIM:-64}
NHEADS=${NHEADS:-8}
ACTOR_LR=${ACTOR_LR:-0.001}
PRIOR_FRACTION=${PRIOR_FRACTION:-0.25}

PREP_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/data/preprocessing.py --dataset gaussian_30 --raw-root /Users/xiaoyuhe/Causal-LLM/dataset/raw --processed-root /Users/xiaoyuhe/Causal-LLM/dataset/processed"
RUN_CMD="${PYTHON_EXEC} /Users/xiaoyuhe/Causal-LLM/scripts/baseline/GUIDE/main.py --data_path /Users/xiaoyuhe/Causal-LLM/dataset/processed/gaussian_30/X.npy --adj_path /Users/xiaoyuhe/Causal-LLM/dataset/processed/gaussian_30/adj.npy --output_dir ${OUTPUT_DIR} --num_epochs ${NUM_EPOCHS} --hidden_dim ${HIDDEN_DIM} --nheads ${NHEADS} --actor_lr ${ACTOR_LR} --prior_fraction ${PRIOR_FRACTION}"

mkdir -p "${OUTPUT_DIR}"

${PREP_CMD}
${RUN_CMD}
