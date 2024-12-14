#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 00:01:00
#SBATCH -o vectorAdd-output.rxt
srun vectorAdd ${1} ${2} ${3} > vectorAdd-output.txt
