#!/bin/bash

#SBATCH -q inference
#SBATCH --job-name=msuHPCDemo
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:1
#SBATCH --mem=16G

module load singularity

singularity exec --nv ../msUNET/tensorflow2003.sif python3 ../msUNET/segment_brain.py -i ../test_dataset/ 
