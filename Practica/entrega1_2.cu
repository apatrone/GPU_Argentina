/*Realizar un programa CUDA que dado un vector V de N números enteros multiplique a 
cada número por una constante C, se deben realizar dos implementaciones:
a.Tanto C como N deben ser pasados como parámetros al kernel.
b.Tanto C como N deben estar almacenados en la memoria de constantes de la GPU*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//M and N number of threads (grid and block)




     
__global__ void addAll( const int array[] , int dim,float result[], const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	//printf("sum:%i\n",  result[0]);
	if(index<dim){
		if(dim<=thread_number){ //if more threads than array size
		//	printf("Thread %i; Adding value of index %i\n", index, index, array[index]);
			atomicAdd(result,array[index]);
		}
		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
			//		printf("Thread %i; Adding value of index %i\n", index, i, array[i]);
					atomicAdd(result,array[i]);
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim/thread_number); i< dim; i++){
				//	printf("Thread %i; Adding value of index %i\n",index, i, array[i]);
					atomicAdd(result,array[i]);
				}
			}
		}
		//printf("sum:%i\n",  result[0]);
	}
	
} 
__global__ void sigma( const int array[] , int dim,float result[], const float mean, const int thread_number)
{
	 int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	//printf("sum:%i\n",  result[0]);
	if(index<dim){
		if(dim<=thread_number){ //if more threads than array size
			//printf("Thread %i; Adding value of index %i\n", index, index, array[index]);
			atomicAdd(result,(array[index]-mean)*(array[index]-mean));
		}
		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
					//printf("Thread %i; Adding value of index %i\n", index, i, array[i]);
					atomicAdd(result,(array[i]-mean)*(array[i]-mean));
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim/thread_number); i< dim; i++){
					//printf("Thread %i; Adding value of index %i\n",index, i, array[i]);
					atomicAdd(result,(array[i]-mean)*(array[i]-mean));
				}
			}
		}
		//printf("sum:%i\n",  result[0]);
	}


}


    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
	time_begin=clock();
    // pointers to host & device arrays
     int *device_array = 0;
     int *host_array = 0;
	 int size_array=9;
	 float *d_sum=NULL;
	 float *h_sum= 0;
	 float mean;
	 float final_res;
	 int M=1, N=1;
	 if(argc == 4){
		 size_array=atoi(argv[1]);
		 N=atoi(argv[2]);
		 M=atoi(argv[3]);
	 }
	 h_sum=( float*)malloc(sizeof( float));
	 h_sum[0]=0;
      // malloc a host array
     host_array = (int*)malloc( size_array * sizeof(int));
	
    for(int i=0; i<size_array; i++){
        host_array[i]=rand()%10;
       // printf("%i\t", host_array[i]);
    }
    printf("\n");
	
	
     // cudaMalloc a device array
     cudaMalloc(&device_array,size_array * sizeof(int));    
	 cudaError_t er=cudaMalloc(&d_sum, sizeof(float));  
    // download and inspect the result on the host:
    cudaError_t e=cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice); 
	cudaError_t error=cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);
	//cudaerrorinvalidvalue(11)

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N*M*M;
    addAll<<<grid, bloque>>>(device_array, size_array , d_sum, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
   //cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost); 
   
    printf("Sum of array: %f\n", h_sum[0]);
	mean=h_sum[0]/size_array;
	h_sum[0]=0;

	cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyHostToDevice);
	sigma<<<grid, bloque>>>(device_array, size_array , d_sum, mean, thread_number);
	cudaThreadSynchronize();
	cudaMemcpy(h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost); 

	 printf("Sigma: %f\n", h_sum[0]);
	
	final_res = sqrt(h_sum[0]/(size_array-1));
	
	printf("Final result: %f\n", final_res);
     // deallocate memory
      free(host_array);free(h_sum);
      cudaFree(device_array); cudaFree(d_sum);

	  printf("Time elapsed: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.215s

}