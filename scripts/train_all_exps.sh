#!/bin/bash

#SBATCH --job-name=maal_all_exps
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node2
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --output=job_output_all_exps.txt
#SBATCH --error=job_error_all_exps.txt

module load Python3.10
source $HOME/env_maal/bin/activate
cd $HOME/MAAL
export CUDA_LAUNCH_BLOCKING=1

echo "Start time: $(date)"

echo "==================== RUNNING EXP A ===================="
python -m src.models.maal_expA --checkpoint-dir checkpoints/expA

echo "==================== RUNNING EXP B ===================="
python -m src.models.maal_expB --checkpoint-dir checkpoints/expB

echo "==================== RUNNING EXP C ===================="
python -m src.models.maal_expC --checkpoint-dir checkpoints/expC

echo "==================== RUNNING EXP D ===================="
python -m src.models.maal_expD --checkpoint-dir checkpoints/expD

echo "==================== RUNNING EXP E ===================="
python -m src.models.maal_expE --checkpoint-dir checkpoints/expE

echo "==================== RUNNING EXP F ===================="
python -m src.models.maal_expF --checkpoint-dir checkpoints/expF

echo "Finish time: $(date)"
