#!/bin/bash

#SBATCH --job-name=agg_fldas
#SBATCH --output=agg_fldas.out
#SBATCH --error=agg_fldas.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/conda-env-haynes/
python /home/haynes13/code/prevented-planting/fldas_xagg.py
