#!/bin/bash

#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100
#SBATCH -t 02:00:00  # Solicita 2 horas para las ejecuciones
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

# Programa a ejecutar
programa="./matrizMul-simple"

# Configuraciones para las pruebas
filas_matrices=(32 256 512 1024 2048)         # Tamaños de filas de A
columnas_matrices=(32 256 512 1024 2048)      # Tamaños de columnas de B
hilos_por_bloque=(1 8 16 32 64 128 256)     
hilos_por_bloque=(1 8 16 32 64 128 256) # Hilos por bloque (múltiplos del warp)

# Itera sobre los diferentes tamaños de matrices y configuraciones
for filasA in "${filas_matrices[@]}"; do
    for columnasB in "${columnas_matrices[@]}"; do
        for columnasA in "${columnas_matrices[@]}"; do
            for hilos in "${hilos_por_bloque[@]}"; do
                for hilos2 in "${hilos2_por_bloque[@]}"; do
                    # Imprime en el archivo los parámetros de la prueba
                    #echo "Prueba: FilasA=$filasA, ColumnasA=$columnasA, ColumnasB=$columnasB, Hilos=$hilos" >> "$archivo_resultados"
                    
                    # Ejecuta el programa con los parámetros actuales
                    srun "$programa" "$filasA" "$columnasB" "$columnasA" "$hilos" "$hilos2" "$archivo_resultados"
    
                    # Separador entre pruebas en el archivo de resultados
                    #echo "----------------------------------------------------" >> "$archivo_resultados"
                done
            done
        done
    done
done

