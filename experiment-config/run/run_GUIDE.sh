#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun"
# CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun --config-name=config-cluster"
${CMD} experiment='GUIDE_experiment' problem='sachs,lucas,hepar2,gaussian_30,gaussian_50,non_gaussian_30,non_gaussian_50' problem.generator='GUIDE'
