#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 00:05:00
#SBATCH -o vectorAdd-output.txt

# Lista de tamaños de vector (en elementos) (Los tamaños son aproximados)
# Tenemos 1M (1 048 576 elementos) = 4MB
# Tenemos 10M (10 485 760) = 40MB
# Tenemos 100M (100 000 000) = 400MB
# Tenemos 1000M (1 000 000 000) = 4GB
VECTOR_SIZES=("1048576" "10485760" "100000000" "1000000000")
THREADS_PER_BLOCK=256
REPS=1

# Ejecutar vectorAdd para cada tamaño de vector
for SIZE in "${VECTOR_SIZES[@]}"; do
    echo "Ejecutando con tamaño de vector: ${SIZE}, threads por bloque: ${THREADS_PER_BLOCK}, repeticiones: ${REPS}"
    srun vectorAdd ${SIZE} ${THREADS_PER_BLOCK} ${REPS} >> vectorAdd-output.txt
done
