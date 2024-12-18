#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 00:05:00
#SBATCH -o vectorAdd-threads-output.txt

# Tamaño fijo del vector (1,000,000,000 elementos)
VECTOR_SIZE=1000000000
REPS=1

# Lista de threads por bloque
THREADS_PER_BLOCK_LIST=("32" "64" "128" "256" "512" "1024")

# Ejecutar vectorAdd para cada valor de threads por bloque
for TPB in "${THREADS_PER_BLOCK_LIST[@]}"; do
    echo "Ejecutando con threads por bloque: ${TPB}, tamaño de vector: ${VECTOR_SIZE}, repeticiones: ${REPS}"
    srun vectorAdd ${VECTOR_SIZE} ${TPB} ${REPS} >> vectorAdd-threads-output.txt
done
