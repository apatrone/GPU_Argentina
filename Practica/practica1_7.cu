
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//M and N number of threads (grid and block)
#define M 1 
#define N 1   



     
__global__ void multiply( const char string[] , const char substring[], const int dim_str,const int dim_substr,  int pos[], const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	bool gg=false;
	if(index<dim_str){
		if(dim_str<=thread_number){ //if more threads than array size
			printf("Thread %i; Modifying value of index %i \n", index, index);
			int j;
			for( j=index; j<dim_str; j++){
				if(string[j] != substring[j] )
					break;
				else if(j==dim_substr-1)
					gg=true;
			}
			if(gg==true){
				pos[0]=index;
				return;
			}

		}

		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim_str/thread_number); i< index*(int)(dim_str/thread_number)+(int)(dim_str/thread_number); i++){
					printf("Thread %i; Modifying value of index %i \n", index, i);
					int j;
					for( j=i; j<dim_str; j++){
						if(string[j] != substring[j] )
							break;
						else if(j==dim_substr-1)
							gg=true;
					}
					if(gg==true){
						pos[0]=i;
						return;
					}
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim_str/thread_number); i< dim_str; i++){
					printf("Thread %i; Modifying value of index %i\n",index, i);
					int j;
					for( j=i; j<dim_str; j++){
						if(string[j] != substring[j] )
							break;
						else if(j==dim_substr-1)
							gg=true;
					}
					if(gg==true){
						pos[0]=i;
						return;
					}
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
      char *device_array = 0;
      char *host_array = "aglolsbdrc";
	  char *d_substr; cudaMalloc(&d_substr,3 * sizeof(char));   
	  char *substr="lol";
	  cudaMemcpy(d_substr, substr, sizeof(char)*3, cudaMemcpyHostToDevice);  
	  int *h_pos, *d_pos;
	 
	  h_pos=( int*)malloc(sizeof( int));
	 cudaError_t er= cudaMalloc(&d_pos, sizeof(int));  

	  int size_array=10;
      // malloc a host array
     // host_array = (char*)malloc( size_array * sizeof(char));
	 // host_array="aglolsbdrc";
    for(int i=0; i<size_array; i++){
        //host_array[i]=rand()%26+52;
        printf("%c\t", host_array[i]);
    }
    printf("\n");

      // cudaMalloc a device array
     cudaError_t err= cudaMalloc(&device_array,size_array * sizeof(char));    
    // download and inspect the result on the host:
    cudaError_t erro=cudaMemcpy(device_array, host_array, sizeof(char)*size_array, cudaMemcpyHostToDevice);         

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N*M*M;
    multiply<<<grid, bloque>>>(device_array, d_substr, size_array ,3, d_pos, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaError_t error=cudaMemcpy(h_pos, d_pos, sizeof(int), cudaMemcpyDeviceToHost); 


     printf("pos: %i\t", h_pos[0]);

	
     // deallocate memory
     // free(host_array); 
	  free(h_pos);
      cudaFree(device_array);
	  cudaFree(d_pos);

	  printf("Time elapsed: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.215s

}