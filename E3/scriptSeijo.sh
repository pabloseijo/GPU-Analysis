#!/bin/bash

#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 00:59:00  # Solicita 59 minutos para las ejecuciones
#SBATCH -o salida_experimentos.txt  # Archivo de salida SLURM
#SBATCH -e errores_experimentos.txt  # Archivo de errores SLURM

# Verifica si el archivo de resultados fue proporcionado como argumento
if [ $# -lt 1 ]; then
    echo "Uso: $0 <archivo_resultados>"
    exit 1
fi

# Archivo de resultados
archivo_resultados=$1

# Asegúrate de que el archivo esté vacío antes de comenzar
> "$archivo_resultados"

# Fuente y programa GPU
fuenteGPU="matrices3.cu"
programaGPU="./ejecutableGPU"

# Configuraciones fijas
filasA=10000
columnasA=10000
columnasB=10000

# Configuración de hilos
hilos_max=32

# Compila el programa CUDA
echo "Compilando el programa CUDA..."
nvcc "$fuenteGPU" -o "$programaGPU"
if [ $? -ne 0 ]; then
    echo "Error al compilar $fuenteGPU"
    exit 1
fi
echo "Compilación exitosa."

# Ejecuta los bucles con los valores establecidos
echo "Iniciando pruebas..."
for hilos in $(seq 1 $hilos_max); do
    # Ejecuta el programa en la GPU
    srun "$programaGPU" "$filasA" "$columnasA" "$columnasB" "$hilos" "$hilos" "$archivo_resultados"
done
echo "Pruebas completadas."
