/**
 * Suma dos vectores: C = A + B (Memoria Unificada).
 */
#include <stdio.h>
#include <time.h>

#define checkError(ans) { asserError((ans), __FILE__, __LINE__); }
inline void asserError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define TSET(time)  clock_gettime( CLOCK_MONOTONIC, &(time) )
#define TINT(ts,te) { ( (double) 1000.*( (te).tv_sec - (ts).tv_sec ) + ( (te).tv_nsec - (ts).tv_nsec )/(double) 1.e6 ) }

#define MAX_TH_PER_BLOCK 1024
#define NELDEF 1000
#define TPBDEF 256
#define NREPDEF 1

typedef float basetype;

/**
 * Código CUDA
 */
__global__ void vectorAdd(const basetype *A, const basetype *B, basetype *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

/**
 * Funcion main en el host
 * Parametros: nElementos threadsPerBlock nreps
 */
int main(int argc, char *argv[]) {
    basetype *A, *B, *C;  // Memoria unificada
    unsigned int numElements = (argc > 1) ? atoi(argv[1]) : NELDEF;
    unsigned int tpb = (argc > 2) ? atoi(argv[2]) : TPBDEF;
    unsigned int nreps = (argc > 3) ? atoi(argv[3]) : NREPDEF;
    size_t size = numElements * sizeof(basetype);
    struct timespec tstart, tend;
    double tint;

    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(tpb);
    dim3 blocksPerGrid((numElements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    printf("Suma de vectores de %u elementos (%u reps), con %u bloques de %u threads\n",
           numElements, nreps, blocksPerGrid.x, threadsPerBlock.x);

    // Reserva de memoria unificada
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // Inicialización de vectores
    for (int i = 0; i < numElements; ++i) {
        A[i] = rand() / (basetype)RAND_MAX;
        B[i] = rand() / (basetype)RAND_MAX;
    }

    // CPU: Suma de vectores
    TSET(tstart);
    for (unsigned int r = 0; r < nreps; ++r) {
        for (int i = 0; i < numElements; ++i) {
            C[i] = A[i] + B[i];
        }
    }
    TSET(tend);
    tint = TINT(tstart, tend);
    printf("HOST: Tiempo para %u sumas de vectores de tamaño %u: %.2f ms\n", nreps, numElements, tint);

    // GPU: Ejecución del kernel
    cudaEventRecord(start, 0);
    for (unsigned int r = 0; r < nreps; ++r) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, numElements);
    }
    cudaDeviceSynchronize(); // Sincronización para garantizar que el kernel ha terminado
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU: Tiempo de ejecución del kernel: %.2f ms\n", elapsedTime);

    // Verificación de resultados
    for (int i = 0; i < numElements; ++i) {
        if (fabs(C[i] - (A[i] + B[i])) > 1e-5) {
            fprintf(stderr, "Error en la verificación del elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Suma correcta.\n");

    // Liberar memoria
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Terminamos\n");
    return 0;
}
