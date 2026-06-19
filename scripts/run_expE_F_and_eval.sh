#!/bin/bash

#SBATCH --job-name=maal_E_F
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node2
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --output=job_output_E_F.txt
#SBATCH --error=job_error_E_F.txt

module load Python3.10
source $HOME/env_maal/bin/activate
cd $HOME/MAAL
export CUDA_LAUNCH_BLOCKING=1

echo "Start time: $(date)"

echo "==================== RUNNING EXP E ===================="
python -m src.models.maal_expE --checkpoint-dir checkpoints/expE

echo "==================== EVALUATING EXP E ===================="
python src/utils/evaluate_maal.py \
    --checkpoint checkpoints/expE/maal/fold_0/best.pt \
    --model-module src.models.maal_expE \
    --num-xai-samples 50


echo "==================== RUNNING EXP F ===================="
python -m src.models.maal_expF --checkpoint-dir checkpoints/expF

echo "==================== EVALUATING EXP F ===================="
python src/utils/evaluate_maal.py \
    --checkpoint checkpoints/expF/maal/fold_0/best.pt \
    --model-module src.models.maal_expF \
    --num-xai-samples 50

echo "Finish time: $(date)"
