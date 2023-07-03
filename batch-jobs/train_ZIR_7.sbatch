#!/bin/bash

#SBATCH --job-name=train_ZIR_7
#SBATCH --output=train_ZIR_7.out
#SBATCH --error=train_ZIR_7.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/train_ZIR_7.py
