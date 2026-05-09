#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun"
# CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun --config-name=config-cluster"
${CMD} experiment='KCRL_experiment' problem='asia,lucas,sachs,Oxygen-therapy' problem.generator='KCRL'
