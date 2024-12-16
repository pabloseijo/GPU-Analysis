#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define checkError(ans) { asserError((ans), __FILE__, __LINE__); }
inline void asserError(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define TSET(time) clock_gettime(CLOCK_MONOTONIC, &(time))
#define TINT(ts, te) (1000.0 * ((te).tv_sec - (ts).tv_sec) + ((te).tv_nsec - (ts).tv_nsec) / 1.0e6)

// Número máximo de threads por cada dimensión del bloque
#define MAX_TH_PER_BLOCK_DIM 32

// Dimensiones predeterminadas de las matrices
#define MATDIMDEF 1000

// Número de threads por dimensión de bloque por defecto
#define TPBDIMDEF 4

// Tipo de datos
typedef float basetype;

void check_memoria(const unsigned int matrizDim);

/**
 * Multiplica dos matrices en el host
 */
__host__ void h_matrizMul(const basetype* A, const basetype* B, basetype* C,
    unsigned int nFilasA, unsigned int nColumnasA, unsigned int nColumnasB) {
    for (unsigned int i = 0; i < nFilasA; ++i) {
        for (unsigned int j = 0; j < nColumnasB; ++j) {
            basetype sum = 0.0f;
            for (unsigned int k = 0; k < nColumnasA; ++k) {
                sum += A[i * nColumnasA + k] * B[k * nColumnasB + j];
            }
            C[i * nColumnasB + j] = sum;
        }
    }
}

/**
 * Multiplica dos matrices en el device
 */
__global__ void d_matrizMul(const basetype* A, const basetype* B, basetype* C,
    unsigned int nFilasA, unsigned int nColumnasA, unsigned int nColumnasB) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; // Índice de fila
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // Índice de columna

    if (i < nFilasA && j < nColumnasB) {
        basetype sum = 0.0f;
        for (unsigned int k = 0; k < nColumnasA; ++k) {
            sum += A[i * nColumnasA + k] * B[k * nColumnasB + j];
        }
        C[i * nColumnasB + j] = sum;
    }
}

/**
 * Funcion main en el host
 * Parametros: nFilasA nColumnasA nColumnasB threadsPerBlock (las filas de B son triviales pues son iguales a las columnas de A)
 */
int main(int argc, char* argv[]) {
    basetype* h_A = NULL, * h_B = NULL, * h_C = NULL, * h_C2 = NULL;
    basetype* d_A = NULL, * d_B = NULL, * d_C = NULL;
    unsigned int nFilasA = 1, nColumnasA = 1, nColumnasB = 1, tpbdim = 1;
    size_t sizeA = 0, sizeB = 0, sizeC = 0;

    // Valores para la medida de tiempos
    struct timespec tstart, tend;
    double tint;

    // Dimensiones de las matrices
    nFilasA = (argc > 1) ? atoi(argv[1]) : MATDIMDEF;
    nColumnasA = (argc > 2) ? atoi(argv[2]) : MATDIMDEF;
    nColumnasB = (argc > 3) ? atoi(argv[3]) : MATDIMDEF;

    // Tamaños de las matrices
    sizeA = nFilasA * nColumnasA * sizeof(basetype);
    sizeB = nColumnasA * nColumnasB * sizeof(basetype);
    sizeC = nFilasA * nColumnasB * sizeof(basetype);

    // Número de threads por bloque
    tpbdim = (argc > 4) ? atoi(argv[4]) : TPBDIMDEF;
    tpbdim = (tpbdim > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM : tpbdim;

    check_memoria(sizeA + sizeB + sizeC);

    // Configuración de Grid y Bloques
    dim3 threadsPerBlock(tpbdim, tpbdim, 1);
    dim3 blocksPerGrid((nColumnasB + tpbdim - 1) / tpbdim, (nFilasA + tpbdim - 1) / tpbdim, 1);

    printf("Multiplicación de matrices (%ux%u) x (%ux%u) -> (%ux%u)\n", nFilasA, nColumnasA, nColumnasA, nColumnasB, nFilasA, nColumnasB);
    printf("Configuración: %ux%u bloques de %ux%u threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // Reserva memoria en el host
    h_A = (basetype*)malloc(sizeA);
    h_B = (basetype*)malloc(sizeB);
    h_C = (basetype*)malloc(sizeC);
    h_C2 = (basetype*)malloc(sizeC);

    if (!h_A || !h_B || !h_C || !h_C2) {
        fprintf(stderr, "Error reservando memoria en el host\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa las matrices en el host
    for (unsigned int i = 0; i < nFilasA * nColumnasA; ++i) h_A[i] = rand() / (basetype)RAND_MAX;
    for (unsigned int i = 0; i < nColumnasA * nColumnasB; ++i) h_B[i] = rand() / (basetype)RAND_MAX;

    // Multiplicación en el host
    TSET(tstart);
    h_matrizMul(h_A, h_B, h_C, nFilasA, nColumnasA, nColumnasB);
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("Host: Tiempo de multiplicación: %lf ms\n", tint);

    // Reserva memoria en el dispositivo
    checkError(cudaMalloc((void**)&d_A, sizeA));
    checkError(cudaMalloc((void**)&d_B, sizeB));
    checkError(cudaMalloc((void**)&d_C, sizeC));

    // Copia las matrices al dispositivo
    checkError(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    checkError(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Multiplicación en el device
    TSET(tstart);
    d_matrizMul << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, nFilasA, nColumnasA, nColumnasB);
    checkError(cudaPeekAtLastError());
    checkError(cudaDeviceSynchronize());
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("Device: Tiempo de multiplicación: %lf ms\n", tint);

    // Copia el resultado del device al host
    checkError(cudaMemcpy(h_C2, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verificación de resultados
    for (unsigned int i = 0; i < nFilasA * nColumnasB; ++i) {
        if (fabs(h_C[i] - h_C2[i]) > 1e-3) {
            fprintf(stderr, "Verificación de resultados falla en el elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Multiplicación correcta.\n");

    // Libera memoria
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C2);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

/**
 * Verifica la memoria de la GPU
 */
void
check_memoria(const unsigned int numElem)
{
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, 0));

    size_t gmem = prop.totalGlobalMem;
    size_t bytes_arrays = numElem * sizeof(basetype);
    double gib = (double)(1073741824.0);

    printf("GiB ocupados en la GPU: %g GiB, memoria global %g GiB\n", bytes_arrays / gib, gmem / gib);
    if (gmem >= bytes_arrays)
        printf("GiB libres en la GPU: %g GiB\n", (gmem - bytes_arrays) / gib);
    else {
        printf("Los arrays no caben en la memoria de la GPU\n");
        exit(EXIT_FAILURE);
    }
        
}
