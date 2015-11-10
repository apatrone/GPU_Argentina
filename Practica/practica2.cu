#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//M and N number of threads (grid and block)
#define M 1 
#define N 22  

#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}

__global__ void multiply_no_shared( int global_array[] , int dim,  const int c, const int thread_number)
{
		
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	 
	if(idx<thread_number) global_array[idx]*=c;
} 
     
__global__ void multiply( int global_array[] , int dim,  const int c, const int thread_number)
{
   extern __shared__ int shared_a[];
		
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	 
	if(idx<thread_number){
		shared_a[idx]=global_array[idx];
		shared_a[idx]*=c;
		global_array[idx]=shared_a[idx];
	}
} 

    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
	
    // pointers to host & device arrays
      int *device_array = 0;
      int *host_array = 0;
		  int size_array=1;
      // malloc a host array
      host_array = (int*)malloc( size_array * sizeof(int));
		  for(int i=0; i<size_array; i++){
		      host_array[i]=rand()%10;
		    //  printf("%i\t", host_array[i]);
		  }
		  //printf("\n");

      // cudaMalloc a device array
    cudaMalloc(&device_array,size_array * sizeof(int));    
    // download and inspect the result on the host:
    cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice);         

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
		int thread_number= N*N*M*M;
		int shared_mem=sizeof(int);

		time_begin=dwalltime();
    multiply_no_shared<<<grid, bloque>>>(device_array, size_array , 2, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
		printf("GPU time without shared memory: %f seconds\n", dwalltime() - time_begin ); 
		
 		for(int i=0; i<size_array; i++){
		      host_array[i]/=2;
		  }
		cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice);         

		time_begin=dwalltime();
    multiply<<<grid, bloque, shared_mem>>>(device_array, size_array , 2, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
		printf("GPU time with shared memory: %f seconds\n", dwalltime() - time_begin ); 
   
//	for(int i=0; i<size_array; i++)
  //      printf("%i\t", host_array[i]);

	
     // deallocate memory
      free(host_array);
      cudaFree(device_array);


}
