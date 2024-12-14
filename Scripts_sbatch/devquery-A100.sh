#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 32G
#SBATCH --gres gpu:a100
#SBATCH -o devquery-a100.out
#SBATCH -t 00:00:01
srun devquery
