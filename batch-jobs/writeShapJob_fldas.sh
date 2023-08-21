#!/bin/bash

template="#!/bin/bash

#SBATCH --job-name=ShapJob_fldas_\$YEAR
#SBATCH --output=ShapJob_fldas_\$YEAR.out
#SBATCH --error=ShapJob_fldas_\$YEAR.err
#SBATCH --ntasks=1
#SBATCH --partition=broadwl
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haynes13@uchicago.edu

source activate /home/haynes13/.conda/envs/prevplant
python /home/haynes13/code/prevented-planting/getSHAP_fldas.py \$YEAR
"

years=(1996 2006 2016)

for year in "${years[@]}"; do
    script="ShapJob_fldas_${year}.sbatch"
    echo -e "$template" | sed "s/\$YEAR/$year/g" > "$script"
    chmod +x "$script"
    
    # Execute the generated script
    "sbatch $script"
done
