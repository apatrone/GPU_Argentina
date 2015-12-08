
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


double secuential(const double a[] , int dim,bool verbose){
	double mean=0;
	for(int i=0; i<dim;i++){
		mean+=a[i];
	}
	mean=mean/dim;
	if(verbose)printf("cpu mean %f\n", mean);
	double sum=0;
	for(int i=0; i<dim;i++){
		sum+=(a[i]-mean)*(a[i]-mean);
	}
	if(verbose)printf("cpu sigma %f\n", sum);
	return sqrt(sum/(dim-1));

}

     
__global__ void reduccion( double a[], const int dim)
{
	unsigned long int i;
	unsigned long int global_id = blockIdx.x * blockDim.x + threadIdx.x;	

	if (global_id < dim/2) {	
		i = 2 * global_id;
		a[global_id] = a[i] + a[i+1];

	}
} 

__global__ void pre_sigma( double a[], const int dim, const double mean)
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
     double *device_array = 0;
     double *host_array = 0;

	unsigned long int size_array  = (argc > 1)? atoi (argv[1]): 1024;
	unsigned int block_size = (argc > 2)? atoi (argv[2]): 16;	
	bool verbose= (argc>3)? (argv[3][0]=='v'): false;

      // malloc a host array
     host_array = (double*)malloc( size_array * sizeof(double));
	 double *copy_host_array=(double*)malloc( size_array * sizeof(double));
	
    for(unsigned int i=0; i<size_array; i++){
		host_array[i]=rand()%10;
		copy_host_array[i]=host_array[i];
		if(verbose)
			printf("%f\t", host_array[i]);

    }
    printf("\n");	
	
     // cudaMalloc a device array
     cudaMalloc(&device_array,size_array * sizeof(double));    

    // download and inspect the result on the host:
    cudaError_t e=cudaMemcpy(device_array, host_array, sizeof(double)*size_array, cudaMemcpyHostToDevice); 

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
	cudaMemcpy(copy_host_array, device_array, sizeof(double)*size_array, cudaMemcpyDeviceToHost); 
	
	double mean_gpu= copy_host_array[0] / size_array;
	//copy again the original array to the device array which had been modified
	e=cudaMemcpy(device_array, host_array, sizeof(double)*size_array, cudaMemcpyHostToDevice); 

	dim3 bloque2(block_size);	
	dim3 grid2((size_array + bloque2.x - 1) / bloque2.x);			
	
	pre_sigma<<<bloque2, grid2>>>(device_array, size_array, mean_gpu);

	cudaThreadSynchronize();
	cudaMemcpy(copy_host_array, device_array, sizeof(double)*size_array, cudaMemcpyDeviceToHost); 

	if(verbose){
		printf("mean gpu: %f\n", mean_gpu);
		for(unsigned int j=0; j<size_array; j++)
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
	
	cudaMemcpy(copy_host_array, device_array, sizeof(double)*size_array, cudaMemcpyDeviceToHost); 

	if(verbose) printf("gpu sigma %f \n", copy_host_array[0]);
	double final_res= sqrt(copy_host_array[0]/(size_array-1));

	printf("GPU time: %f seconds\t", (((double)clock() - (double)time_begin) / 1000000.0F ) * 1000  ); 
	printf("GPU result: %f\n", final_res);

	//--------------cpu computations-----------------------//
	time_begin=clock();
	double cpu_res=secuential(host_array, size_array, verbose);
	printf("CPU time: %f seconds\t", (((double)clock() - (double)time_begin) / 1000000.0F ) * 1000  );
	printf("CPU result: %f\n", cpu_res);

    // deallocate memory
    free(host_array);free(copy_host_array);
    cudaFree(device_array); 

	

}