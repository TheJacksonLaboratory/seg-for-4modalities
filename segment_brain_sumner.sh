#!/bin/bash

#SBATCH -J msuCPUDemo
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

module load singularity

singularity exec seg-for-4modalities/tensorflow2003.sif python3 -m seg-for-4modalities.segment_brain --input_type dataset --input demo/test_dataset