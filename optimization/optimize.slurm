#!/bin/bash

#############################

# job name (displayed by the queue)
#SBATCH -J readout_optimizer

# Use an array of 100 jobs simultaneously
#SBATCH --array=1-100

# computing partition (defq == default)
#SBATCH -p defq

# CPU cores (5 folds)
#SBATCH -c 5

# Use a singular node
#SBATCH --nodes=1

# Standard error messages are saved in this file
#SBATCH -e logs/slurm_%A_%a.log
#SBATCH -o logs/slurm_%A_%a.log

# Change working directory
#SBATCH --chdir=.

# Recieve email alerts
#SBATCH --mail-type=ALL

#############################

# useful information to print
echo "#############################"
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"

############################
module load Python/3.10

python3.10 -m venv venv
venv/bin/pip install -e .

# run optimization script from virtual environment
venv/bin/python -m optimization.optimize research --type training \
    --trials=10\
    --study_name=reservoir-optimization \
    --job_id=${SLURM_ARRAY_TASK_ID} \
    --processes=1

# all done
echo "Optimization Finished"