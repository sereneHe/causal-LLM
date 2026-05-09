#!/bin/sh

export PYTHONPATH=$PYTHONPATH:../
export KMP_DUPLICATE_LIB_OK=TRUE

CMD="uv run python /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun"
# CMD="uv run python /Users/xiaoyuhe/Causal-LLM/experiment-config/run_experiment.py --multirun --config-name=config-cluster"
${CMD} experiment='ILSCSL_experiment' problem='asia,child' problem.generator='ILSCSL' ilscsl.llm_url="${ILS_CSL_LLM_URL:-}"
