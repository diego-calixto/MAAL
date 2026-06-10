#!/bin/bash

#SBATCH --job-name=maal_pipeline

#SBATCH -p short-complex

#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -c 16

#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt

module load Python3.10

source $HOME/env_maal/bin/activate

cd $HOME/MAAL

python Attention.py