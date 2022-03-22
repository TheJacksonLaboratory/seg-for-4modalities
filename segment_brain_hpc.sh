#!/bin/bash

#SBATCH -q inference
#SBATCH --job-name=msuHPCDemo
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:1
#SBATCH --mem=16G

module load singularity

singularity exec --nv seg-for-4modalities/tensorflow2003.sif python -m seg-for-4modalities.segment_brain --input_type dataset --input demo/test_dataset
