#!/bin/bash
#SBATCH -J xlm-roberta-build-and-run        
#SBATCH --time=24:00:00 
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniil.zhuravel@stud-mail.uni-wuerzburg.de
#SBATCH -p standard
#SBATCH -c 16
#SBATCH --tmp=300G
#SBATCH --output=logs/slurm-%j.out    # %j = job ID
#SBATCH --error=logs/slurm-%j.err

set -e

echo "Job started on $(hostname) at $(date)"

source ~/transfer_project/venv/bin/activate

mkdir -p logs


cd ~/transfer_project

echo "Running main.py" 

python -u main.py

echo "Job finished at $(date)"