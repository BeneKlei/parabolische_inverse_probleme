#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=36          # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
 
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=benedikt.klein@uni-muenster.de # your mail address
 
# LOAD MODULES HERE IF REQUIRED
#module load foss/2019a

#WORKING_DIR="/home/benedikt/Dokumente/parabolische_inverse_probleme"
WORKING_DIR="/home/b/b_klei15/parabolische_inverse_probleme"
INTERPRETER_PATH="venv_39/bin/python"
SCRIPT_PATH="RBInvParam/deployment/run_optimization.py"

"$WORKING_DIR/$INTERPRETER_PATH" "$WORKING_DIR/$SCRIPT_PATH" \
                                 "--setup" $1 \
                                 "--optimizer-parameter" $2 \
                                 "--save-path" $3





