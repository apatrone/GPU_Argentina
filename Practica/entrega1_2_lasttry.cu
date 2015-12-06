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


float secuential(const float a[] , int dim,bool verbose){
	float mean=0;
	for(int i=0; i<dim;i++){
		mean+=a[i];
	}
	mean=mean/dim;
	if(verbose)printf("cpu mean %f\n", mean);
	float sum=0;
	for(int i=0; i<dim;i++){
		sum+=(a[i]-mean)*(a[i]-mean);
	}
	if(verbose)printf("cpu sigma %f\n", sum);
	return sqrt(sum/(dim-1));

}

     
__global__ void reduccion( float a[], const int dim)
{
	unsigned long int i;
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;	

	if (global_id < dim/2) {	
		i = 2 * global_id;
		a[global_id] = a[i] + a[i+1];

	}
} 

__global__ void pre_sigma( float a[], const int dim, const float mean)
{
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;	

	if (global_id < dim) {
		a[global_id] -= mean;
		a[global_id]*= a[global_id];
	}

	
} 

    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
	
    // pointers to host & device arrays
     float *device_array = 0;
     float *host_array = 0;

	unsigned long int size_array  = (argc > 1)? atoi (argv[1]): 1024;
	unsigned int block_size = (argc > 2)? atoi (argv[2]): 16;	
	bool verbose= (argc>3)? (argv[3][0]=='v'): false;

      // malloc a host array
     host_array = (float*)malloc( size_array * sizeof(float));
	 float *copy_host_array=(float*)malloc( size_array * sizeof(float));
	
    for(int i=0; i<size_array; i++){
		host_array[i]=rand()%10;
		copy_host_array[i]=host_array[i];
		if(argc==4 && verbose){
			printf("%f\t", host_array[i]);
		}
		else if(argc==4)
			host_array[i]=atoi(argv[3]);
    }
    printf("\n");	
	
     // cudaMalloc a device array
     cudaMalloc(&device_array,size_array * sizeof(int));    

    // download and inspect the result on the host:
    cudaError_t e=cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice); 

	//cudaerrorinvalidvalue(11)

	time_begin=clock();

	unsigned long int i = size_array;
	while (i > 1) {	
		dim3 bloque(block_size);
		dim3 grid((i/2 + bloque.x - 1)/ bloque.x);			
		reduccion<<<grid, bloque>>>(device_array, i);
		cudaThreadSynchronize();
		i = i/2;
	}

    // download and inspect the result on the host:
	cudaMemcpy(copy_host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
	
	float mean_gpu= copy_host_array[0] / size_array;
	//copy again the original array to the device array which had been modified
	e=cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice); 

	dim3 bloque2(block_size);	
	dim3 grid2((size_array + bloque2.x - 1) / bloque2.x);			
	
	pre_sigma<<<bloque2, grid2>>>(device_array, size_array, mean_gpu);

	cudaThreadSynchronize();
	cudaMemcpy(copy_host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 

	if(verbose){
		printf("mean gpu: %f\n", mean_gpu);
		for(int j=0; j<size_array; j++)
			printf("%f\t", copy_host_array[j]);
		printf("\n");
	}

	i=size_array;
	while (i > 1) {	
		dim3 bloque(block_size);
		dim3 grid((i/2 + bloque.x - 1)/ bloque.x);			
		reduccion<<<grid, bloque>>>(device_array, i);
		cudaThreadSynchronize();
		i = i/2;
	}
	
	cudaMemcpy(copy_host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 

	if(verbose) printf("gpu sigma %f \n", copy_host_array[0]);
	float final_res= sqrt(copy_host_array[0]/(size_array-1));

	printf("GPU time: %f seconds\t", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); 
	printf("GPU result: %f\n", final_res);

	//--------------cpu computations-----------------------//
	time_begin=clock();
	float cpu_res=secuential(host_array, size_array, verbose);
	printf("CPU time: %f seconds\t", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  );
	printf("CPU result: %f\n", cpu_res);

    // deallocate memory
    free(host_array);free(copy_host_array);
    cudaFree(device_array); 

	

}