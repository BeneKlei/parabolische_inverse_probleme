#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=16          # the number cpus per task
##SBATCH --partition=long          # on which partition to submit the job
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)

##SBATCH --mem=128G
#SBATCH --mem=48G
 
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=benedikt.klein@uni-muenster.de # your mail address

#module load palma/2024a GCCcore/13.3.0 Python/3.12.3 foss/2024a 

export LD_LIBRARY_PATH=/home/b/b_klei15/software/dealii_skylake/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Usage:
#   sbatch run_batch.sbatch /path/to/working_dir
# or:
#   sbatch --export=ALL,WORKING_DIR=/path/to/working_dir run_batch.sbatch

python ./problem_batch.py --working-dir "$1" \
                                        > "$1/out.log" \
                                        2> "$1/err.log"