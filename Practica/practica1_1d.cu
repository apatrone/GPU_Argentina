/*Realizar un programa CUDA que dado un vector V de N números enteros multiplique a 
cada número por una constante C, se deben realizar dos implementaciones:
a.Tanto C como N deben ser pasados como parámetros al kernel.
b.Tanto C como N deben estar almacenados en la memoria de constantes de la GPU*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//M and N number of threads (grid and block)
#define M 1 
#define N 1   

int h_array_size=10; //host 
__constant__ int d_array_size; //device
double h_c=2; //host 
__constant__ double d_c; //device

     
__global__ void multiply( double array[] , const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	
	if(index<d_array_size){
		if(d_array_size<=thread_number){ //if more threads than array size
			printf("Thread %i; Modifying value of index %i for %f * %f because < d_array_size %i\n", index, index, array[index], d_c, d_array_size);
			array[index]*=d_c; 
		}
		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with h_array_size/thread_nb array entries
				for(int i=index*(int)(d_array_size/thread_number); i< index*(int)(d_array_size/thread_number)+(int)(d_array_size/thread_number); i++){
					printf("Thread %i; Modifying value of index %i for %f * %f because < d_array_size %i\n", index, i, array[i], d_c, d_array_size);
					array[i]*=d_c; 
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(d_array_size/thread_number); i< d_array_size; i++){
					printf("Thread %i; Modifying value of index %i for %f * %f because < d_array_size %i\n",index, i, array[i], d_c, d_array_size);
					array[i]*=d_c; 
				}
			}
		}
	}
	
} 

    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
	time_begin=clock();
    // pointers to host & device arrays
      double *device_array = 0;
      double *host_array = 0;
	  int h_array_size=10;
      
	  //copy variables from host to device
	  cudaMemcpyToSymbol(d_array_size,&h_array_size,sizeof(h_array_size));
	  cudaMemcpyToSymbol(d_c,&h_c,sizeof(h_c));
	  // malloc a host array
      host_array = (double*)malloc( h_array_size * sizeof(double));

    for(int i=0; i<h_array_size; i++){
        host_array[i]= static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10)); //random float between 0 and 10
        printf("%f\t", host_array[i]);
    }
    printf("\n");

    // cudaMalloc a device array
    cudaMalloc(&device_array,h_array_size * sizeof(double));    
    // download and inspect the result on the host:
    cudaMemcpy(device_array, host_array, sizeof(double)*h_array_size, cudaMemcpyHostToDevice);         

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N*M*M;
    multiply<<<grid, bloque>>>(device_array, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, sizeof(double)*h_array_size, cudaMemcpyDeviceToHost); 

    for(int i=0; i<h_array_size; i++)
        printf("%f\t", host_array[i]);

	printf("Time elapsed: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.081s
     // deallocate memory
      free(host_array);
      cudaFree(device_array);


}