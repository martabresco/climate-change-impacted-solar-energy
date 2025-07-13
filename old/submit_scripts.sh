#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=power no temp regrid
#SBATCH --nodes=2
#SBATCH --mail-user=s233224@dtu.dk
#SBATCH --mail-type=END
#SBATCH --exclusive=user
#SBATCH --partition=rome,workq

module load python/3.13.1            # Load the Python module (adjust based on your HPC environment)
python temp_calc.py                 # Execute your Python script