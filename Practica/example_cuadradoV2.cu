/* 
   Taller De Programación Sobre GPUs (General Purpose Computation on Graphics Processing Unit)

   Adrián Pousa,
   Victoria Sanz. 

   2013

   cuadradoV.cu
   Dado un vector V calcula el cuadrado de cada elemento del mismo, resultando V[i]=V[i]*V[i]. 
   El resultado queda en el vector R. 
   
   Parámetros opcionales (en este orden): 
   #n: número de elementos en cada vector
   #blk: hilos por bloque CUDA
*/

#include <stdio.h>
#include <stdlib.h>

// Tipo de los elementos en los vectores
// Compilar con -D_INT_ para vectores de tipo entero
// Compilar con -D_DOUBLE_ para vectores de tipo double
// Predeterminado vectores de tipo float

#ifdef _INT_
typedef int basetype;     // Tipo para elementos: int
#define labelelem    "ints"
#elif _DOUBLE_
typedef double basetype;  // Tipo para elementos: double
#define labelelem    "doubles"
#else
typedef float basetype;   // Tipo para elementos: float     PREDETERMINADO
#define labelelem    "floats"
#endif

const int N = 1048576;    // Número predeterminado de elementos en los vectores

const int CUDA_BLK = 64;  // Tamaño predeterminado de bloque de hilos CUDA

/* 
   Para medir el tiempo transcurrido (elapsed time):

   resnfo: tipo de dato definido para abstraer la métrica de recursos a usar
   timenfo: tipo de dato definido para abstraer la métrica de tiempo a usar

   timestamp: abstrae función usada para tomar las muestras del tiempo transcurrido

   printtime: abstrae función usada para imprimir el tiempo transcurrido

   void myElapsedtime(resnfo start, resnfo end, timenfo *t): función para obtener 
   el tiempo transcurrido entre dos medidas
*/

#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


/*
  Función para inicializar el vector que vamos a utilizar
*/
void init_CPU_array(basetype array[], const unsigned int n)
{
  unsigned int i;
  for(i = 0; i < n; i++) {
    array[i] = (basetype)i;
  }
}


//  Función que comprueba la que cada elemento en el vector resultado se calculó correctamente
void check_array(basetype array[], const unsigned int n){
  unsigned long int i;
  basetype data;  
  for(i = 0; i < n; i++) {
    data = (basetype)i * (basetype)i;
    if ((basetype)array[i] != data ){
         printf("\n Error posicion %ld - array[i] %f ",i,array[i]);
	}
  }

}

//  Función secuencial: cuadradoV para CPU (*r* veces)
void cuadradoV_CPU(basetype arrayV[], const unsigned int n) {
 unsigned int i;
 for(i = 0; i < n; i++) {
     arrayV[i] = arrayV[i]*arrayV[i];
 }

}

//  Definición de nuestro kernel para función cuadradoV
__global__ void cuadradoV_kernel_cuda(basetype *const arrayV,   const int n){

  unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id < n)
    arrayV[global_id] = arrayV[global_id]*arrayV[global_id];

}

//  Función para sumar dos vectores en la GPU
void cuadradoV_GPU( basetype arrayV[], const unsigned int n, const unsigned int blk_size){
double timetick;

  // Número de bytes de cada uno de nuestros vectores
  unsigned int numBytes = n * sizeof(basetype);

  // Reservamos memoria global del device (GPU) para el array y lo copiamos
  basetype *cV;
  
  timetick = dwalltime();
  cudaMalloc((void **) &cV, numBytes);
  printf("-> Tiempo de alocacion en memoria global de GPU %f\n", dwalltime() - timetick);  
  timetick = dwalltime();
  cudaMemcpy(cV, arrayV, numBytes, cudaMemcpyHostToDevice); // CPU -> GPU
  printf("-> Tiempo de copia de memoria CPU =>> GPU %f\n", dwalltime() - timetick);
  
  // Bloque unidimensional de hilos (*blk_size* hilos)
  dim3 dimBlock(blk_size);

  // Grid unidimensional (*ceil(n/blk_size)* bloques)
  dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

  // Lanzamos ejecución del kernel en la GPU
  //timestamp(start);            // Medimos tiempo de cálculo en GPU
  timetick = dwalltime();
  cuadradoV_kernel_cuda<<<dimGrid, dimBlock>>>(cV, n);
  cudaThreadSynchronize();
  printf("-> Tiempo de ejecucion en GPU %f\n", dwalltime() - timetick);
  //timestamp(end);

  // Movemos resultado: GPU -> CPU
  timetick = dwalltime();
  cudaMemcpy(arrayV, cV, numBytes, cudaMemcpyDeviceToHost); // GPU -> CPU
  printf("-> Tiempo de copia GPU ==>> CPU %f\n", dwalltime() - timetick);
  
  // Liberamos memoria global del device utilizada
  cudaFree (cV);


}

// Declaración de función para ver recursos del device
void devicenfo(void);

// Declaración de función para comprobar y ajustar los parámetros de
// ejecución del kernel a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *cb);

int main(int argc, char *argv[]){

double timetick;

  // Aceptamos algunos parámetros desde línea de comandos

  // Número de elementos del vector (predeterminado: N 1048576)
  unsigned int n = (argc > 1)?atoi (argv[1]):N;

  // Número de hilos en cada bloque CUDA (predeterminado: CUDA_BLK 64)
  unsigned int cb = (argc > 2)?atoi (argv[2]):CUDA_BLK;

  checkparams(&n, &cb);

  // Número de bytes a reservar para nuestro vector
  unsigned int numBytes = n * sizeof(basetype);

  // Reservamos e inicializamos el vector en CPU
  timetick = dwalltime();
  basetype *vectorV = (basetype *) malloc(numBytes); // Vector con datos de entrada
  init_CPU_array(vectorV, n);
  printf("-> Tiempo de alocar memoria e inicializar vectores en CPU %f\n", dwalltime() - timetick);
  
  // Ejecutamos cuadradoV en CPU
  timetick = dwalltime();
  cuadradoV_CPU(vectorV,n);
  printf("-> Tiempo de ejecucion en CPU %f\n", dwalltime() - timetick);

  //Chequea si el resultado obtenido en la CPU es correcto
  check_array(vectorV,n); 

  //Inicializa nuevamente el vector para realizar la ejecucion en GPU
  init_CPU_array(vectorV, n);
  
  // Ejecutamos cuadradoV en GPU
  cuadradoV_GPU(vectorV, n,  cb);

  //Chequea si el resultado obtenido en la GPU es correcto
  check_array(vectorV,n); 
  
  free(vectorV);

  return(0);
}


//  Sacar por pantalla información del *device*
void devicenfo(void){
  struct cudaDeviceProp capabilities;

  cudaGetDeviceProperties (&capabilities, 0);

  printf("->CUDA Platform & Capabilities\n");
  printf("Name: %s\n", capabilities.name);
  printf("totalGlobalMem: %.2f MB\n", capabilities.totalGlobalMem/1024.0f/1024.0f);
  printf("sharedMemPerBlock: %.2f KB\n", capabilities.sharedMemPerBlock/1024.0f);
  printf("regsPerBlock (32 bits): %d\n", capabilities.regsPerBlock);
  printf("warpSize: %d\n", capabilities.warpSize);
  printf("memPitch: %.2f KB\n", capabilities.memPitch/1024.0f);
  printf("maxThreadsPerBlock: %d\n", capabilities.maxThreadsPerBlock);
  printf("maxThreadsDim: %d x %d x %d\n", capabilities.maxThreadsDim[0], 
	 capabilities.maxThreadsDim[1], capabilities.maxThreadsDim[2]);
  printf("maxGridSize: %d x %d\n", capabilities.maxGridSize[0], 
	 capabilities.maxGridSize[1]);
  printf("totalConstMem: %.2f KB\n", capabilities.totalConstMem/1024.0f);
  printf("major.minor: %d.%d\n", capabilities.major, capabilities.minor);
  printf("clockRate: %.2f MHz\n", capabilities.clockRate/1024.0f);
  printf("textureAlignment: %d\n", capabilities.textureAlignment);
  printf("deviceOverlap: %d\n", capabilities.deviceOverlap);
  printf("multiProcessorCount: %d\n", capabilities.multiProcessorCount);
}


//  Función que ajusta el número de hilos, de bloques, y de bloques por hilo 
//  de acuerdo a las restricciones de la GPU
void checkparams(unsigned int *n, unsigned int *cb){
  struct cudaDeviceProp capabilities;

  // Si menos numero total de hilos que tamaño bloque, reducimos bloque
  if (*cb > *n)
    *cb = *n;

  cudaGetDeviceProperties (&capabilities, 0);

  if (*cb > capabilities.maxThreadsDim[0]) {
    *cb = capabilities.maxThreadsDim[0];
    printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n\n", 
	   *cb);
  }

  if (((*n + *cb - 1) / *cb) > capabilities.maxGridSize[0]) {
    *cb = 2 * (*n - 1) / (capabilities.maxGridSize[0] - 1);
    if (*cb > capabilities.maxThreadsDim[0]) {
      *cb = capabilities.maxThreadsDim[0];
      printf("->Núm. hilos/bloq cambiado a %d (máx por bloque para dev)\n", 
	     *cb);
      if (*n > (capabilities.maxGridSize[0] * *cb)) {
	*n = capabilities.maxGridSize[0] * *cb;
	printf("->Núm. total de hilos cambiado a %d (máx por grid para \
dev)\n\n", *n);
      } else {
	printf("\n");
      }
    } else {
      printf("->Núm. hilos/bloq cambiado a %d (%d máx. bloq/grid para \
dev)\n\n", 
	     *cb, capabilities.maxGridSize[0]);
    }
  }
}

