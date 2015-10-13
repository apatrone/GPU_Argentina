
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//M and N number of threads (grid and block)

void secuential(const int a[] ,const int b[], const int sqrt_dim);
     
__global__ void multiply( const int a[] ,const int b[], int c[] , const int sqrt_dim,const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	//for an element in matrix[i][j] , its coordinate k in array[] is i+j*sqrt(size_array)
	int index_i = index < sqrt_dim ? index : (int)index%sqrt_dim; 
   	int index_j = (index-index_i)/sqrt_dim;
	int dim=sqrt_dim*sqrt_dim;

	//printf("index= %i \t", index); 

	if(index<dim){
		c[index]=0;
		if(dim<=thread_number){ //if more threads than array size
			//printf("Thread %i; Modifying value of index %i\n ", index, index);
			c[index]= b[index]; //c= b
			c[index]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
			for(int i=0; i<sqrt_dim;i++){ //row of first matrix	
				c[index]+=a[i+index_j * sqrt_dim ]*b[i + index_i*sqrt_dim]; //c+= a*b^t
			}
			
		}
		else{ //if less threads than array size
				
				if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
					for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
					//	printf("Thread %i; Modifying value of index %i \n", index, i);
						index_i =  (int)i%sqrt_dim; 
						index_j = (i-index_i)/sqrt_dim;
						c[i]= b[i]; //c= b
						c[i]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
						for(int j=0; j<sqrt_dim;j++){ //row of first matrix
							c[i]+=a[j+index_j * sqrt_dim ]*b[j+ index_i*sqrt_dim]; //c+= a*b^t
						} 
					}
				}
				else{ //if last thread deal with all remaining array entries
					for(int i=index*(int)(dim/thread_number); i< dim; i++){
			//			printf("Thread %i; Modifying value of index %i\n",index, i );
						index_i = (int)i%sqrt_dim; 
						index_j = (i-index_i)/sqrt_dim;
						c[i]= b[i]; //c= b
						c[i]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
						for(int j=0;j<sqrt_dim;j++){ //row of first matrix
							c[i]+=a[j+index_j * sqrt_dim ]*b[j + index_i*sqrt_dim]; //c+= a*b^t
						}
					}
				}
			}
		}
	
} 

    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
    // pointers to host & device arrays
      int *d_array1 = 0,*d_array2 = 0,*d_array3 = 0;
      int *h_array1 = 0,*h_array2 = 0,*h_array3 = 0;
	  int size_array=9; //here, size_array =L hqs to be a square

	 int M=1, N=1;
	 if(argc == 4){
		 size_array=atoi(argv[1]);
		 N=atoi(argv[2]);
		 M=atoi(argv[3]);
	 }  
      // malloc columns of host arrays
      h_array1 = (int*)malloc( size_array * sizeof(int));
	  h_array2 = (int*)malloc( size_array * sizeof(int));
	  h_array3 = (int*)malloc( size_array * sizeof(int));
	
	/*	  
	printf("Array A:\n");
	for(int i=0; i<size_array; i++){
		h_array1[i]=rand()%10;
		printf("%i\t",  h_array1[i]);
		if((i+1)%(int)sqrt((float)size_array)==0)
			printf("\n");
	}
	printf("\n");
 
	printf("Array B:\n");
	for(int i=0; i<size_array; i++){
		h_array2[i]=rand()%10;
		printf("%i\t",  h_array2[i]);
		if((i+1)%(int)sqrt((float)size_array)==0)
			printf("\n");
	}
	printf("\n");*/
 

     // cudaMalloc a device array
    cudaMalloc(&d_array1,size_array * sizeof(int));    
	cudaMalloc(&d_array2,size_array * sizeof(int));  
	cudaMalloc(&d_array3,size_array * sizeof(int));  
    // download and inspect the result on the host:
    cudaMemcpy(d_array1, h_array1, sizeof(int)*size_array, cudaMemcpyHostToDevice);   
	cudaMemcpy(d_array2, h_array2, sizeof(int)*size_array, cudaMemcpyHostToDevice);   

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos (max 512 threads in a block)
    dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N*M*M;
	time_begin=clock();
    multiply<<<grid, bloque>>>(d_array1, d_array2 , d_array3,sqrt((float)size_array), thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(h_array3, d_array3, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
	
	printf("GPU time, %i threads: %f seconds\n", thread_number,(((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s

/*	printf("Array C=B + AB^t + A^t :\n");
    for(int i=0; i<size_array; i++){
        printf("%i\t", h_array3[i]);
	if((i+1)%(int)(sqrt((float)size_array))==0)
		printf("\n");
	}
	printf("\n");*/
	time_begin=clock();
	secuential(h_array1, h_array2, sqrt((float)size_array));
	printf("CPU time: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s
     // deallocate memory
    free(h_array3); free(h_array2); free(h_array1);
    cudaFree(d_array3);cudaFree(d_array2);cudaFree(d_array1);

	  

}

void secuential(const int a[] ,const int b[], const int sqrt_dim){
	int dim = sqrt_dim* sqrt_dim;
	int index_i, index_j;
	int *c= (int *)malloc ( dim * sizeof(int));
	for(int i=0; i< dim; i++){
		index_i = (int)i%sqrt_dim; 
		index_j = (i-index_i)/sqrt_dim;
		c[i]= b[i]; //c= b
		c[i]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
		for(int j=0;j<sqrt_dim;j++){ //row of first matrix
			c[i]+=a[j+index_j * sqrt_dim ]*b[j + index_i*sqrt_dim]; //c+= a*b^t
		}
	}

	/*printf("Sequential result: Array C=B + AB^t + A^t :\n");
    for(int i=0; i<dim; i++){
        printf("%i\t", c[i]);
		if((i+1)%(int)(sqrt((float)dim))==0)
			printf("\n");
	}
	printf("\n");*/
	free(c);
}