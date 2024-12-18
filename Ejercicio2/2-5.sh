#!/bin/sh
#SBATCH -n 1                  # Número de tareas (1 nodo)
#SBATCH -c 32                 # Número de cores por tarea
#SBATCH --mem 64G             # Memoria solicitada
#SBATCH --gres gpu:a100       # Solicitar una GPU A100
#SBATCH -t 00:15:00           # Tiempo límite (15 minutos)
#SBATCH -o vectorAddEj4-output.txt  # Archivo de salida

# Nombre del archivo fuente y ejecutable
SOURCE="vectorAddEj4.cu"
EXECUTABLE="vectorAddEj4"

# Parámetros del programa
VECTOR_SIZE=100000000  # Número de elementos en el vector (10^8)
THREADS_PER_BLOCK=256  # Número de threads por bloque
REPS=10                # Número de repeticiones

# 1. Compilar el código CUDA
echo "Compilando el archivo ${SOURCE}..."
nvcc -arch=sm_80 ${SOURCE} -o ${EXECUTABLE}
if [ $? -ne 0 ]; then
    echo "Error: Fallo en la compilación de ${SOURCE}."
    exit 1
fi
echo "Compilación exitosa."

# 2. Ejecutar el programa compilado
echo "Ejecutando ${EXECUTABLE} con los siguientes parámetros:"
echo "Tamaño del vector: ${VECTOR_SIZE}, Threads por bloque: ${THREADS_PER_BLOCK}, Repeticiones: ${REPS}"
srun ./${EXECUTABLE} ${VECTOR_SIZE} ${THREADS_PER_BLOCK} ${REPS}

echo "Ejecución completada. Revisa vectorAddEj5-output.txt para los resultados."
