#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun"
# CMD="python3 /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun --config-name=config-cluster"
${CMD} experiment='UnifiedOneShot_experiment' problem='asia,alarm,sachs,Hepar2,Lucas,dream41,dream42,dream43,dream44,BIAS,LEGAL' problem.generator='UnifiedOneShot'
