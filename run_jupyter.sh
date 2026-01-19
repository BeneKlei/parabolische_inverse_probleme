source ./venv/bin/activate &&
python -m jupyter lab --ip=0.0.0.0 --no-browser --allow-root

python ../RBInvParam/deployment/run_experiment_batch.py ./elasticity_compare.py palma --working_dir /scratch/tmp/b_klei15/experiments/

export CMAKE_ARGS="-Ddeal.II_DIR=/home/b/b_klei15/software/dealii_skylake/lib/cmake/deal.II"

salloc --nodes 1 --cpus-per-task 36 --time 00:30:00 --constraint=skylake --partition=express
module load palma/2024a GCCcore/13.3.0 Python/3.12.3 foss/2024a 
export LD_LIBRARY_PATH="/home/b/b_klei15/software/dealii_skylake/lib/:$LD_LIBRARY_PATH"
source ../venv/bin/activate 

squeue -u $USER -t RUNNING -o "%i %j" \
  | grep 'new_baseline_noise_level_' \
  | awk '{print $1}' \
  | xargs scancel


#export LD_LIBRARY_PATH="/home/dealii/workdir/RBInvParam/problems/shared/pymor_dealii_bindings:$LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH="/home/benedikt/Dokumente/parabolische_inverse_probleme/RBInvParam/problems/shared/pymor_dealii_bindings:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/b/b_klei15/parabolische_inverse_probleme/RBInvParam/problems/shared/pymor_dealii_bindings:$LD_LIBRARY_PATH"