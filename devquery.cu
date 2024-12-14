#include <stdio.h>

void printDevProp(cudaDeviceProp devProp)
{
// TODO: Completar esta función
}

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
