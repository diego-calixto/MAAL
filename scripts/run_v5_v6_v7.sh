#!/bin/bash

#SBATCH --job-name=maal_I_J_K
#SBATCH -p short-complex
#SBATCH --nodelist=cluster-node9
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --output=job_output_I_J_K.txt
#SBATCH --error=job_error_I_J_K.txt

module load Python3.10
source $HOME/maal/bin/activate
cd $HOME/MAAL
export CUDA_LAUNCH_BLOCKING=1

echo "Start time: $(date)"

echo "==================== RUNNING MAAL V5 (EXP I) ===================="
python -m src.models.maal_expI --checkpoint-dir resultados_cluster/maal_v5/checkpoints

echo "==================== RUNNING MAAL V6 (EXP J) ===================="
python -m src.models.maal_expJ --checkpoint-dir resultados_cluster/maal_v6/checkpoints

echo "==================== RUNNING MAAL V7 (EXP K) ===================="
python -m src.models.maal_expK --checkpoint-dir resultados_cluster/maal_v7/checkpoints

echo "==================== EVALUATING MAAL V5, V6, V7 ===================="
python scripts/evaluate_all_metrics.py MAAL_V5 MAAL_V6 MAAL_V7

echo "Finish time: $(date)"
