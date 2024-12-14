#include <stdio.h>

void printDevProp(cudaDeviceProp devProp);

int main(int argc, char *argv[]) {
  int numDevs;
  cudaDeviceProp prop;
  cudaError_t error;

  // Obtiene el número de dispositivos (tarjetas GPUs disponibles)
  error = cudaGetDeviceCount(&numDevs);
  if(error != cudaSuccess) {
    fprintf(stderr, "Error obteniendo numero de dispositivos: %s en %s linea %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }

  printf("Numero de dispositivos = %d\n", numDevs);

  // Recorre las tarjetas disponibles y obtiene las propiedades de las mismas en prop.
  for(int i=0; i < numDevs; i++) {
    error = cudaGetDeviceProperties(&prop, i);
    if(error != cudaSuccess) {
      fprintf(stderr, "Error obteniendo propiedades del dispositivo %d: %s en %s linea %d\n", i, cudaGetErrorString(error), __FILE__, __LINE__);
      exit(EXIT_FAILURE);
    }
    printf("\nDispositivo #%d\n", i);
    printDevProp(prop);
  }

  return(EXIT_SUCCESS);
}

void printDevProp(cudaDeviceProp devProp) {
  
    printf("Device Name: %s\n", devProp.name);

    // 1. Capacidad de cómputo
    printf("Compute Capability: %d.%d\n", devProp.major, devProp.minor);

    // 2. Número de multiprocesadores (SMs)
    printf("Number of SMs: %d\n", devProp.multiProcessorCount);

    // 3. Máximo número de hilos residentes por SM
    printf("Max Threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);

    // 4. Tamaño máximo de cada dimensión de un grid
    printf("Max Grid Size: [%d, %d, %d]\n", 
           devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

    // 5. Máximo número de hilos por bloque
    printf("Max Threads per Block: %d\n", devProp.maxThreadsPerBlock);

    // 6. Tamaño máximo de cada dimensión de un bloque
    printf("Max Block Size: [%d, %d, %d]\n", 
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);

    // 7. Número de registros de 32 bits disponibles
    printf("Registers per Block: %d\n", devProp.regsPerBlock);
    printf("Registers per SM: %d\n", devProp.regsPerBlock * devProp.multiProcessorCount);

    // 8. Memoria compartida disponible
    printf("Shared Memory per Block: %.2f KB\n", devProp.sharedMemPerBlock / 1024.0);
    printf("Shared Memory per SM: %.2f KB\n", 
           (devProp.sharedMemPerBlock * devProp.multiProcessorCount) / 1024.0);

    // 9. Memoria global
    printf("Global Memory: %.2f GB\n", devProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // 10. Frecuencia pico de la memoria (en Hz) y ancho del bus
    printf("Memory Clock Rate: %.2f Hz\n", devProp.memoryClockRate * 1000.0);
    printf("Memory Bus Width: %d bits\n", devProp.memoryBusWidth);

    // 11. Ancho de banda pico
    double BWpeak = (devProp.memoryClockRate * 1000.0 * (devProp.memoryBusWidth / 8.0) * 2) / 1e9;
    printf("Peak Memory Bandwidth: %.2f GiB/s\n", BWpeak);

    // 12. Número total de CUDA cores
    int coresPerSM = 0;
    switch (devProp.major) {
        case 8:  // Ampere (A100)
            coresPerSM = 128;  // 128 cores por SM
            break;
        case 7:  // Turing (T4)
            coresPerSM = 64;   // 64 cores por SM
            break;
        default:
            coresPerSM = 0;  // Desconocido
    }
    int totalCores = coresPerSM * devProp.multiProcessorCount;
    printf("Total CUDA Cores: %d\n", totalCores);
}
