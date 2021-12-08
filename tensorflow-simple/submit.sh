#!/bin/bash

#SBATCH --job-name=tensoflow_test
#SBATCH -o _test.out
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --ntasks=20
#SBATCH --mem=50G
#SBATCH --gres=gpu:1

# Use the new module system
module use /home/software/tools/eb_modulefiles/all/Core

#to load tf 2.6.0 you'll first need the compiler set it was built with
module load foss/2021a

#load tf
module load TensorFlow/2.6.0-CUDA-11.3.1

# Run the simple tensorflow example - taken from the docs: https://www.tensorflow.org/tutorials/quickstart/beginner
python example.py
