#!/bin/bash

WORKING_DIR="/home/benedikt/Dokumente/parabolische_inverse_probleme"
INTERPRETER_PATH="venv/bin/python"
SCRIPT_PATH="RBInvParam/deployment/run_optimization.py"

"$WORKING_DIR/$INTERPRETER_PATH" "$WORKING_DIR/$SCRIPT_PATH" \
                                 "--setup" $1 \
                                 "--optimizer-parameter" $2 \
                                 "--save-path" $3




