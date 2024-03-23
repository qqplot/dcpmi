#!/bin/bash

#SBATCH --job-name=hypers
#SBATCH --partition=P2
#SBATCH --nodes=1    
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=8
#SBATCH --output=./slrum_logs/S-%x.%j.out     

eval "$(conda shell.bash hook)"
conda activate dcpmi

# Beam 
# srun python run.py --output_file output_beam.json --gpu_id 0 --run_type beam

# CPMI
# srun python run.py --output_file output_cpmi.json --gpu_id 0 --use_cpmi --run_type cpmi 

# Ours
# srun python run.py --output_file output_ours.json --gpu_id 0 --use_cpmi --run_type ours --alpha 0.6 --beta 0.01 --use_language_model --soft_uncertainty_weight


# Search hyperparameters
srun python search_hyperparameters.py --output_file ours.csv --gpu_id 0 --use_cpmi --run_type ours --use_language_model --soft_uncertainty_weight