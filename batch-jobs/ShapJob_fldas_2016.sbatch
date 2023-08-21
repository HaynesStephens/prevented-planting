#!/bin/bash

#SBATCH --job-name=ShapJob_fldas_2016
#SBATCH --output=ShapJob_fldas_2016.out
#SBATCH --error=ShapJob_fldas_2016.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/getSHAP_fldas.py 2016

