#!/bin/sh

# Deprecated compatibility wrapper.
#
# Use the consolidated launcher instead:
#   sh /Users/xiaoyuhe/Causal-LLM/experiment-config/run/run_LLM_MINOBSx.sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)

exec sh "$REPO_ROOT/experiment-config/run/run_LLM_MINOBSx.sh" "$@"
