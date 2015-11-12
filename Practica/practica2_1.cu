#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


/*
#include <sys/time.h>
#include <sys/resource.h>

double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}*/

__global__ void multiply_no_shared( int global_array[] , int dim,  const int c, const int tile_width)
{
		
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	 
	if(idx<dim) global_array[idx]*=c;
} 
     
__global__ void multiply( int global_array[] , int dim,  const int c, const int tile_width)
{
   extern __shared__ int shared_a[];
		
	int idx = blockIdx.x*blockDim.x+threadIdx.x;	 
	if(idx<dim){
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
	
	  unsigned int size_array=16;
	 bool verbose=false;
	 int tile_width =16;
	 if(argc == 3){
		 size_array=atoi(argv[1]) ;
		 tile_width=atoi(argv[2]);
	
	 }
	 else if(argc==4){
		 size_array=atoi(argv[1]);
		 tile_width=atoi(argv[2]);
		 verbose=(argv[3][0]=='v');
	 }
      // malloc a host array
      host_array = (int*)malloc( size_array * sizeof(int));
		  for(int i=0; i<size_array; i++){
		      host_array[i]=rand()%10;
		    if(verbose) printf("%i\t", host_array[i]);
		  }
		  if(verbose) printf("\n");

      // cudaMalloc a device array
    cudaMalloc(&device_array,size_array * sizeof(int));    
    // download and inspect the result on the host:
    cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice);         

   	dim3 bloque(tile_width, tile_width);
	dim3 grid((int)ceil(double((float)size_array)/double(bloque.x)), ceil(double((float)size_array)/double(bloque.y)));

	printf("%i threads per block, %i vector\n", tile_width*tile_width,  size_array);
	
	int shared_mem=sizeof(int);

	time_begin=clock();	//time_begin=dwalltime();
    multiply_no_shared<<<grid, bloque>>>(device_array, size_array , 2, tile_width);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
		//printf("GPU time without shared memory: %f seconds\n", dwalltime() - time_begin ); 
		printf("GPU time without shared memory: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  );
 		for(int i=0; i<size_array; i++){
		      host_array[i]/=2;
		  }
		cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice);         

	time_begin=clock();	//	time_begin=dwalltime();
    multiply<<<grid, bloque, shared_mem>>>(device_array, size_array , 2, tile_width);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
		//printf("GPU time with shared memory: %f seconds\n", dwalltime() - time_begin ); 
   printf("GPU time with shared memory: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  );
	if(verbose){
	   for(int i=0; i<size_array; i++)
			printf("%i\t", host_array[i]);
	}
	
     // deallocate memory
      free(host_array);
      cudaFree(device_array);


}
