#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 32G
#SBATCH --gres gpu:t4
#SBATCH -o devquery-t4.out
#SBATCH -t 00:00:01
srun devquery
