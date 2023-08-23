#!/bin/bash

#SBATCH --job-name=trainZIR_230822
#SBATCH --output=trainZIR_230822.out
#SBATCH --error=trainZIR_230822.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/trainZIR_230822.py
