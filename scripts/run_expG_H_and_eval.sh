#!/bin/bash

#SBATCH --job-name=maal_G_H
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node9
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --output=job_output_G_H.txt
#SBATCH --error=job_error_G_H.txt

module load Python3.10
source $HOME/maal/bin/activate
cd $HOME/MAAL
export CUDA_LAUNCH_BLOCKING=1

echo "Start time: $(date)"

echo "==================== RUNNING EXP G ===================="
python -m src.models.maal_expG --checkpoint-dir checkpoints/expG

echo "==================== EVALUATING EXP G ===================="
python src/utils/evaluate_maal.py \
    --checkpoint checkpoints/expG/maal/fold_0/best.pt \
    --model-module src.models.maal_expG \
    --num-xai-samples 50


echo "==================== RUNNING EXP H ===================="
python -m src.models.maal_expH --checkpoint-dir checkpoints/expH

echo "==================== EVALUATING EXP H ===================="
python src/utils/evaluate_maal.py \
    --checkpoint checkpoints/expH/maal/fold_0/best.pt \
    --model-module src.models.maal_expH \
    --num-xai-samples 50

echo "Finish time: $(date)"
