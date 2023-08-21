#!/bin/bash

#SBATCH --job-name=ShapJob_fldas_1996
#SBATCH --output=ShapJob_fldas_1996.out
#SBATCH --error=ShapJob_fldas_1996.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/getSHAP_fldas.py 1996

