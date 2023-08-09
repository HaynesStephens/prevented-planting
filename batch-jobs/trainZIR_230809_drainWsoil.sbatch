#!/bin/bash

#SBATCH --job-name=trainZIR_230809_drainWsoil
#SBATCH --output=trainZIR_230809_drainWsoil.out
#SBATCH --error=trainZIR_230809_drainWsoil.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/trainZIR_230809_drainWsoil.py
