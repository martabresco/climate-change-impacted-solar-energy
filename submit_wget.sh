#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=HadGEM3-GC31-MM
#SBATCH --nodes=1
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

cd /groups/FutureWind/SFCRAD/HadGEM3-GC31-MM/ssp585/r1i1p1f3

bash /groups/FutureWind/SFCRAD/HadGEM3-GC31-MM/ssp585/r1i1p1f3/wget_script_2025-3-5_16-44-17.sh -H
