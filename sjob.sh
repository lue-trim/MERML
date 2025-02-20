#!/bin/bash

## Slurm environment
#SBATCH --job-name MERML
#SBATCH --time 60:00:00
#SBATCH --partition h800_batch
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
##SBATCH --nodelist dgx-h800-03
#SBATCH --mem 20G
##SBATCH --mem-per-cpu 1G
##SBATCH --output 150_output.txt

## source
source ~/.bashrc

cd ~/MERML/Dushman\ reaction

micromamba run -n MERML python main.py
