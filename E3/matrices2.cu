
/**
 * Multiplica dos matrices cuadradas: C = A * B.
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
 
 // Numero maximo de threads por cada dimensión del bloque
 // Consideramos threadsPerBlock.x == threadsPerBlock.y
 //
 #define MAX_TH_PER_BLOCK_DIM 32
 
 // Tamanho por defecto de las matrices
 #define MATDIMDEF 1000
 
 // Numero de threads por cada dimensión bloque por defecto
 #define TPBDIMDEF 4
 
 // Tipo de datos
 typedef float basetype;
 
 void check_memoria(const unsigned int matrizDim);
 
 /**
  * Codigo host
  */
 __host__ void
 h_matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int matrizDim)
 {
   for (unsigned int i = 0; i < matrizDim; ++i)
     for (unsigned int j = 0; j < matrizDim; ++j) {
       basetype sum = (basetype) 0.0;
       for (unsigned int k = 0; k < matrizDim; ++k)
         sum += A[i*matrizDim + k]*B[k*matrizDim + j];
       C[i*matrizDim + j] = sum;
   }
 }
 
 /**
  * Codigo CUDA
  * Cada thread computa un elemento de C
  */
 __global__ void
 matrizMul(const basetype *A, const basetype *B, basetype *C, unsigned int nFilasA, unsigned int nColumnasA, unsigned int nColumnasB)
 {
   // TODO: Calcula el indice de la fila de C y A
   int i = blockIdx.y * blockDim.y + threadIdx.y;
   // TODO Calcula el indice de la columna de C y B
   int j = blockIdx.x * blockDim.x + threadIdx.x;;
 
   if ((i < nFilasA) && (j < nColumnasB))
   {
     basetype sum = (basetype) 0.0;
     for(unsigned int k = 0; k < matrizDim; ++k)
     {
        sum += A[i * nColumnasA + k] * B[k * nColumnasB + j];
     }
     C[i * nColumnasB + j] = sum;   
   }
 }
 
 /**
  * Funcion main en el host
  * Parametros: nElementos threadsPerBlock
  */
 int
 main(int argc, char *argv[])
 {
   basetype *h_A=NULL, *h_B=NULL, *h_C=NULL, *h_C2=NULL;
   basetype *d_A=NULL, *d_B=NULL, *d_C=NULL;
   unsigned int nFilasA = 1, nColumnasA = 1, nColumnasB  = 1, tpbdim = 1, numElem = 1;
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
 
   // Numero de threads por cada dimension  del bloque
   tpbdim = (argc > 2) ? atoi(argv[2]):TPBDIMDEF;
   // Comprueba si es superior al máximo
   tpbdim = (tpbdim > MAX_TH_PER_BLOCK_DIM) ? MAX_TH_PER_BLOCK_DIM:tpbdim;
 
   check_memoria( sizeA+sizeB+sizeC );
 
   // Caracteristicas del Grid
   // Hilos por bloque: primer parámetro dim_x, segundo dim_y
   dim3 threadsPerBlock( tpbdim, tpbdim, 1 );
   // TODO: Calcula el número de bloques en el Grid (bidimensional)
   dim3 blocksPerGrid((nColumnasB + tpbdim - 1) / tpbdim, (nFilasA + tpbdim - 1) / tpbdim, 1);

 
    printf("Multiplicación de matrices (%ux%u) x (%ux%u) -> (%ux%u)\n", nFilasA, nColumnasA, nColumnasA, nColumnasB, nFilasA, nColumnasB);
    printf("Configuración: %ux%u bloques de %ux%u threads\n", blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

   h_A = (basetype *) malloc(size);
   h_B = (basetype *) malloc(size);
   h_C = (basetype *) malloc(size);
   h_C2 = (basetype *) malloc(size);
 
   // Comprueba errores
   if (h_A == NULL || h_B == NULL || h_C == NULL)
   {
     fprintf(stderr, "Error reservando memoria en el host\n");
     exit(EXIT_FAILURE);
   }
 
   // Inicializa las matrices en el host
for (unsigned int i = 0; i < nFilasA * nColumnasA; ++i) h_A[i] = rand() / (basetype)RAND_MAX;
for (unsigned int i = 0; i < nColumnasA * nColumnasB; ++i) h_B[i] = rand() / (basetype)RAND_MAX;
 
   // Inicio tiempo
   TSET(tstart);
   //clock_gettime( CLOCK_MONOTONIC, &tstart );
   // Multiplica las matrices en el host
   h_matrizMul( h_A, h_B, h_C, matrizDim );
   // Fin tiempo
   TSET( tend );
   tint = TINT(tstart, tend);
   printf( "HOST: Tiempo multiplicacion: %lf ms\n", tint );
 
   // Inicio tiempo multiplicacion GPU
   TSET( tstart );
 
   // Reserva memoria para las matrices en el dispositivo
   checkError( cudaMalloc((void **) &d_A, size) );
   checkError( cudaMalloc((void **) &d_B, size) );
   checkError( cudaMalloc((void **) &d_C, size) );
 
   // Copia las matrices h_A y h_B del host al dispositivo
   checkError( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
   checkError( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );
 
   // TODO: Lanza el kernel CUDA
   matrizMul<<< blocksPerGrid, threadsPerBlock >>>( d_A, d_B, d_C, nFilasA, nColumnasA, nColumnasB );
 
   // Comprueba si hubo un error al el lanzamiento del kernel
   // Notar que el lanzamiento del kernel es asíncrono por lo que
   // este chequeo podría no detectar errores en la ejecución del mismo
   checkError( cudaPeekAtLastError() );
   // Sincroniza los hilos del kernel y chequea errores
   // Este chequeo detecta posibles errores en la ejecución
   // Notar que esta sincrinización puede degradar el rendimiento
   checkError( cudaDeviceSynchronize() );
 
   // Copia el vector resultado del dispositivo al host
   checkError( cudaMemcpy(h_C2, d_C, size, cudaMemcpyDeviceToHost) );
 
   // Fin tiempo multiplicacion GPU
   TSET( tend );
   // Calcula tiempo para la multiplicacion GPU
   tint = TINT(tstart, tend);
   printf( "DEVICE: Tiempo multiplicacion: %lf ms\n", tint );
 
 
  // Verificación de resultados
for (unsigned int i = 0; i < nFilasA * nColumnasB; ++i) {
    if (fabs(h_C[i] - h_C2[i]) > 1e-3) {
        fprintf(stderr, "Verificación de resultados falla en el elemento %d!\n", i);
        exit(EXIT_FAILURE);
    }
}
 
   printf("Multiplicacion correcta.\n");
 
   // Liberamos la memoria del dispositivo
   checkError( cudaFree(d_A) );
   checkError( cudaFree(d_B) );
   checkError( cudaFree(d_C) );
 
   // Liberamos la memoria del host
   free(h_A);
   free(h_B);
   free(h_C);
 
   printf("Terminamos\n");
   return 0;
 }
 
 void
 check_memoria(const unsigned int numElem)
 {
   cudaDeviceProp prop;
   checkError( cudaGetDeviceProperties(&prop, 0) );
 
   size_t gmem = prop.totalGlobalMem;
   size_t bytes_arrays = numElem*sizeof(basetype);
   double gib = (double)(1073741824.0);
 
   printf( "GiB ocupados en la GPU: %g GiB, memoria global %g GiB\n", bytes_arrays/gib, gmem/gib );
   if( gmem >= bytes_arrays )
     printf( "GiB libres en la GPU: %g GiB\n", (gmem-bytes_arrays)/gib );
   else
     printf( "Los arrays no caben en la memoria de la GPU\n" );
 }
 
