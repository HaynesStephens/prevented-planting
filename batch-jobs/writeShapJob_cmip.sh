#!/bin/bash

cmip_model="BCC-CSM2-MR"

template="#!/bin/bash

#SBATCH --job-name=ShapJob_\$cmip_model\_\$YEAR
#SBATCH --output=ShapJob_\$cmip_model\_\$YEAR.out
#SBATCH --error=ShapJob_\$cmip_model\_\$YEAR.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/getSHAP_cmip.py \$cmip_model \$YEAR
"

years=(2023)

for year in "${years[@]}"; do
    script="ShapJob_fldas_${year}.sbatch"
    echo -e "$template" | sed -e "s/\$YEAR/$year/g" -e "s/\$CMIP_MODEL/$cmip_model/g" > "$script"
    chmod +x "$script"
    
    # Execute the generated script
    # "sbatch $script"
done