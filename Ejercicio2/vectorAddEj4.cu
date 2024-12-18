/**
 * Suma dos vectores: C = A + B (Medición de tiempos por etapas)
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

int main(int argc, char *argv[]) {
    basetype *h_A = NULL, *h_B = NULL, *h_C = NULL, *h_C2 = NULL;
    basetype *d_A = NULL, *d_B = NULL, *d_C = NULL;
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

    // --- Reserva de memoria en el host ---
    h_A = (basetype *)malloc(size);
    h_B = (basetype *)malloc(size);
    h_C = (basetype *)malloc(size);
    h_C2 = (basetype *)malloc(size);

    // Inicialización de vectores
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (basetype)RAND_MAX;
        h_B[i] = rand() / (basetype)RAND_MAX;
    }

    // --- (a) Reserva de memoria en la GPU ---
    cudaEventRecord(start, 0);
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU: Tiempo de reserva de memoria: %.2f ms\n", elapsedTime);

    // --- (b) Copia de vectores Host -> Device ---
    cudaEventRecord(start, 0);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU: Tiempo de copia Host -> Device: %.2f ms\n", elapsedTime);

    // --- (c) Ejecución del kernel ---
    cudaEventRecord(start, 0);
    for (unsigned int r = 0; r < nreps; ++r) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU: Tiempo de ejecución del kernel: %.2f ms\n", elapsedTime);

    // --- (d) Copia de resultados Device -> Host ---
    cudaEventRecord(start, 0);
    cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU: Tiempo de copia Device -> Host: %.2f ms\n", elapsedTime);

    // Verificación de resultados
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_C2[i] - (h_A[i] + h_B[i])) > 1e-5) {
            fprintf(stderr, "Error en la verificación del elemento %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Suma correcta.\n");

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Terminamos\n");
    return 0;
}
