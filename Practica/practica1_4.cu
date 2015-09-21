/* número por una constante C, se deben realizar dos implementaciones:
a.Tanto C como N deben ser pasados como parámetros al kernel.
b.Tanto C como N deben estar almacenados en la memoria de constantes de la GPU*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//M and N number of threads (grid and block)
#define M 1 
#define N 10   



     
__global__ void multiply( const int a[] , int c[] , const int sqrt_dim,const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	
//for an element in matrix[i][j] , its coordinate k in array[] is i+j*sqrt(size_array) 
//transposed index is j+i*sqrt(size) 
//to get i and j from an index: i = (int)index/sqrt(size) and j=(index-i)/sqrt(size)
	int index_i = index < sqrt_dim ? index : (int)index%sqrt_dim; 
   	int index_j = (index-index_i)/sqrt_dim;
	int dim=sqrt_dim*sqrt_dim;
	
	if(index<dim){
	printf("index %i in matrix form would be i=%i and j=%i\n", index, index_i, index_j);
		if(dim<=thread_number){ //if more threads than array size
			printf("Thread %i; Modifying value of index %i\n ", index, index);
			c[index_j+index_i * sqrt_dim]=a[index]; 
			
		}
	else{ //if less threads than array size
			printf("index %i in matrix form would be i=%i and j=%i\n", index, index_i, index_j);
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
					printf("Thread %i; Modifying value of index %i \n", index, i);
					c[index_j+index_i * sqrt_dim]=a[i]; 
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim/thread_number); i< dim; i++){
					printf("Thread %i; Modifying value of index %i\n",index, i );
					c[index_j+index_i *sqrt_dim]=a[i]; 
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
      int *d_array1 = 0,*d_array2 = 0;
      int *h_array1 = 0,*h_array2 = 0;
	  int size_array=16; //here, size_array =L hqs to be a square
      // malloc columns of host arrays
      h_array1 = (int*)malloc( size_array * sizeof(int));
	h_array2 = (int*)malloc( size_array * sizeof(int));

	  
		  
 
	for(int i=0; i<size_array; i++){
		h_array1[i]=rand()%10;
		printf("%i\t",  h_array1[i]);
		if((i+1)%(int)(sqrt((float)size_array))==0)
			printf("\n");
	}
	printf("\n");
 

     // cudaMalloc a device array
    cudaMalloc(&d_array1,size_array * sizeof(int));    
	cudaMalloc(&d_array2,size_array * sizeof(int));  

    // download and inspect the result on the host:
    cudaMemcpy(d_array1, h_array1, sizeof(int)*size_array, cudaMemcpyHostToDevice);   


    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N*M*M;
    multiply<<<grid, bloque>>>(d_array1, d_array2 , sqrt((float)size_array), thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(h_array2, d_array2, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 

    for(int i=0; i<size_array; i++){
        printf("%i\t", h_array2[i]);
	if((i+1)%(int)(sqrt((float)size_array))==0)
		printf("\n");
	}
	printf("\n");	
     // deallocate memory
     free(h_array2); free(h_array1);
     cudaFree(d_array2);cudaFree(d_array1);

	  printf("Time elapsed: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.2s
}
