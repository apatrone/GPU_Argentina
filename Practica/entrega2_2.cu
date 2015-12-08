// This example demonstrates a parallel sum reduction
// using two kernel launches
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <time.h>

double secuential(const double a[] , int dim,bool verbose){
	double mean=0;
	for(int i=0; i<dim;i++){
		mean+=a[i];
	}
	if(verbose)printf("cpu sum %f\t", mean);
	mean=mean/dim;
	if(verbose)printf("cpu mean %f\t", mean);
	double sum=0;
	for(int i=0; i<dim;i++){
		sum+=(a[i]-mean)*(a[i]-mean);
	}
	if(verbose)printf("cpu sigma %f\n", sum);
	return sqrt(sum/(dim-1));

}

__global__ void reduccion(const double *a,double *a_out,const size_t dim)
{
  extern __shared__ double shared[];

  unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  shared[threadIdx.x]= (global_id < dim) ? a[global_id]: 0;
  __syncthreads();

  for(int i = blockDim.x / 2; i > 0; i /= 2)
  {
    if(threadIdx.x < i)
      shared[threadIdx.x] += shared[threadIdx.x + i];

    __syncthreads();
  }

  if(threadIdx.x == 0)
    a_out[blockIdx.x] = shared[0];
}

__global__ void pre_sigma( double a[], const int dim, const double mean)
{
	/*extern __shared__ double shared[];
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;	
	shared[threadIdx.x]= (global_id < dim) ? a[global_id]: 0;
	__syncthreads();

	//if ( global_id< dim) {
		shared[threadIdx.x] -= mean;
		shared[threadIdx.x]*= shared[threadIdx.x];
	//}

	__syncthreads();
	 a[global_id] = shared[threadIdx.x];*/
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;	

	if (global_id < dim) {
		a[global_id] -= mean;
		a[global_id]*= a[global_id];
	}
} 

int main(int argc, char *argv[])
{
 clock_t time_begin;
 unsigned long int size_array  = (argc > 1)? atoi (argv[1]): 4;
 unsigned int block_size = (argc > 2)? atoi (argv[2]): 2;	
 bool verbose= (argc>3)? (argv[3][0]=='v'): false;
  // generate random input on the host
  double *host_array=(double*)malloc( size_array * sizeof(double));
  for(unsigned int i = 0; i < size_array; ++i)
  {
    host_array[i] =rand()%10;
	if(verbose) printf("%f\t", host_array[i]);
  }
  if(verbose) printf("\n");

 // for(unsigned int i=0; i< size_array; i++)
	//	host_result+= host_array[i];
  time_begin=clock();
  double cpu_result=secuential(host_array, size_array,verbose);
  printf("cpu result: %f\n", cpu_result);
  // move input to device memory
  double *device_array = 0;
  cudaMalloc((void**)&device_array, sizeof(double) * size_array);
  cudaMemcpy(device_array, &host_array[0], sizeof(double) * size_array, cudaMemcpyHostToDevice);

  const size_t bloques = (size_array/block_size) + ((size_array%block_size) ? 1 : 0);

  double *device_array_out = 0;
  cudaMalloc((void**)&device_array_out, sizeof(double) * (bloques + 1));

  reduccion<<<bloques,block_size,block_size * sizeof(double)>>>(device_array, device_array_out, size_array);

  reduccion<<<1,bloques,bloques * sizeof(double)>>>(device_array_out, device_array_out + bloques, bloques);

  // copy the result back to the host
  double device_result = 0;
  cudaMemcpy(&device_result, device_array_out + bloques, sizeof(double), cudaMemcpyDeviceToHost);

   printf("gpu sum: %f\t", device_result);
   double gpu_mean=device_result /size_array;
   printf("gpu mean: %f\n", gpu_mean);
   //---------------
   dim3 bloque2(block_size);	
   dim3 grid2((size_array + bloque2.x - 1) / bloque2.x);			
	
   pre_sigma<<<bloque2, grid2>>>(device_array, size_array, gpu_mean);
   cudaThreadSynchronize();
   cudaMemcpy(host_array, device_array, sizeof(double)*size_array, cudaMemcpyDeviceToHost); 
   if(verbose){
	   for(unsigned int j=0; j<size_array; j++)
			printf("%f\t", host_array[j]);
		printf("\n");
   }


   //------------
   reduccion<<<bloques,block_size,block_size * sizeof(double)>>>(device_array, device_array_out, size_array);

  reduccion<<<1,bloques,bloques * sizeof(double)>>>(device_array_out, device_array_out + bloques, bloques);

   cudaMemcpy(&device_result, device_array_out + bloques, sizeof(double), cudaMemcpyDeviceToHost);
    printf("gpu sigma: %f\n", device_result);
	double final_res= sqrt(device_result/(size_array-1));
	 printf("gpu result: %f\n", final_res);
  // deallocate device memory
  cudaFree(device_array);
  cudaFree(device_array_out);

  return 0;
}

