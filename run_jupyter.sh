source ./venv/bin/activate &&
python -m jupyter lab --ip=0.0.0.0 --no-browser --allow-root

python ../RBInvParam/deployment/run_experiment_batch.py ./elasticity_compare.py palma --working_dir /scratch/tmp/b_klei15/experiments/

module load palma/2024a GCCcore/13.3.0 Python/3.12.3 foss/2024a 
