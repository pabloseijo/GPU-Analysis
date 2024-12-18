#!/bin/sh
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 10:00:00
#SBATCH -o vectorAdd-reps-output.txt

# Parámetros fijos
VECTOR_SIZE=100000000
# Para el número de threads utilizamos el mejor obtenido en el apartado anterior
THREADS_PER_BLOCK=256

# Lista de repeticiones del lazo
REPS_LIST=("1" "10" "100" "500" "1000")

# Ejecutar vectorAdd para cada número de repeticiones
for REPS in "${REPS_LIST[@]}"; do
    echo "Ejecutando con repeticiones: ${REPS}, tamaño de vector: ${VECTOR_SIZE}, threads por bloque: ${THREADS_PER_BLOCK}"
    srun vectorAdd ${VECTOR_SIZE} ${THREADS_PER_BLOCK} ${REPS} >> vectorAdd-reps-output.txt
done
