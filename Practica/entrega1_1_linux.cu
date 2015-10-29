#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>
#include <sys/resource.h>
double dwalltime(){
        double sec;
        struct timeval tv;

        gettimeofday(&tv,NULL);
        sec = tv.tv_sec + tv.tv_usec/1000000.0;
        return sec;
}


void secuential(const int a[] ,const int b[], unsigned long int c[], const int sqrt_dim);
     
__global__ void multiply( const int a[] ,const int b[], unsigned long int c[] , const int sqrt_dim,const int thread_number)
{
   
	unsigned long int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;
	unsigned long int index = blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;

 	//convert global index to column and row (index_i and index_j) of matrix
	unsigned long int index_i = index < sqrt_dim ? index : (int)index%sqrt_dim; 
   	unsigned long int index_j = (index-index_i)/sqrt_dim;

	unsigned int dim=sqrt_dim*sqrt_dim;

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
					for(unsigned long int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
						//printf("Thread %i; Modifying value of index %i \n", index, i);
						index_i =  (int)i%sqrt_dim; 
						index_j = (i-index_i)/sqrt_dim;
						c[i]= b[i]; //c= b
						c[i]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
						
						for(unsigned long int j=0; j<sqrt_dim;j++){ //row of first matrix
							c[i]+=a[j+index_j * sqrt_dim ]*b[j+ index_i*sqrt_dim]; //c+= a*b^t
						} 
					}
				}
				else{ //if last thread deal with all remaining array entries
					for(unsigned long int i=index*(int)(dim/thread_number); i< dim; i++){
						//printf("Thread %i; Modifying value of index %i\n",index, i );
						index_i = (int)i%sqrt_dim; 
						index_j = (i-index_i)/sqrt_dim;
						c[i]= b[i]; //c= b
						c[i]+= a[index_j+ index_i * sqrt_dim]; //c+= a^t
						for(unsigned long int j=0;j<sqrt_dim;j++){ //row of first matrix
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
      int *d_array1 = 0,*d_array2 = 0; unsigned long int *d_array3 = 0;
      int *h_array1 = 0,*h_array2 = 0;unsigned long int*h_array3 = 0;
	  unsigned long int *h_array_sec= 0;
	 unsigned int size_array=191*191; //here, size_array =L has to be a square
	 bool verbose=false;
	 int N=1;
	 if(argc == 3){
		 size_array=atoi(argv[1]) * atoi(argv[1]) ;
		 N=atoi(argv[2]);
	
	 }  
	 else if(argc==4){
		 size_array=atoi(argv[1]) * atoi(argv[1]) ;
		 N=atoi(argv[2]);
		 verbose=(argv[3][0]=='v');

	 }
      // malloc columns of host arrays
	  h_array1 = (int*)malloc( size_array * sizeof(int));
	  h_array_sec= (unsigned long int*)malloc( size_array * sizeof(unsigned long int));
	  h_array2 = (int*)malloc( size_array * sizeof(int));
	  h_array3 = (unsigned long int*)malloc( size_array * sizeof(unsigned long int));
	
	  
	//printf("Array A:\n");
	for(unsigned long int i=0; i<size_array; i++){
		h_array1[i]=1;//rand()%10;
	//	printf("%i\t",  h_array1[i]);
		//if((i+1)%(int)sqrt((float)size_array)==0)
		//	printf("\n");
	}
	//printf("\n");
 
	//printf("Array B:\n");
	for(unsigned int i=0; i<size_array; i++){
		h_array2[i]=1;//rand()%10; 	
		//printf("%i\t",  h_array2[i]);
		//if((i+1)%(int)sqrt((float)size_array)==0)
		//	printf("\n");
	}
	//printf("\n");
 

     // cudaMalloc a device array
    cudaMalloc(&d_array1,size_array * sizeof(int));    
	cudaMalloc(&d_array2,size_array * sizeof(int));  
	cudaMalloc(&d_array3,size_array * sizeof(unsigned long int));  
    // download and inspect the result on the host:
    cudaMemcpy(d_array1, h_array1, sizeof(int)*size_array, cudaMemcpyHostToDevice);   
	cudaMemcpy(d_array2, h_array2, sizeof(int)*size_array, cudaMemcpyHostToDevice);   

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos (max 512 threads in a block)
    dim3 grid(1,1);  //Grid bidimensional de M*M bloques
	int thread_number= N*N;
	if (N*N > 512){
            bloque.x = 512;
            bloque.y = 512;
            grid.x = ceil(double(N)/double(bloque.x));
            grid.y = ceil(double(N)/double(bloque.y));
       }
	printf("%i threads, %ix%i matrix\n", thread_number,  (int)sqrt((float)size_array), (int)sqrt((float)size_array));
	time_begin=dwalltime();
    
	multiply<<<grid, bloque>>>(d_array1, d_array2 , d_array3,sqrt((float)size_array), thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(h_array3, d_array3, sizeof(unsigned long int)*size_array, cudaMemcpyDeviceToHost); 
	
	//printf("GPU time: %f seconds\n", dwalltime() - time_begin);
	//windows time
	printf("GPU time, %i threads: %f seconds\n", thread_number,dwalltime() - time_begin ); 


	if(verbose){
		printf("Array C=B + AB^t + A^t :\n");
		for(int i=0; i<size_array; i++){
			printf("%i\t", h_array3[i]);
		if((i+1)%(int)(sqrt((float)size_array))==0)
			printf("\n");
		}
		printf("\n");
	}
	time_begin=dwalltime();
	secuential(h_array1, h_array2,  h_array_sec, sqrt((float)size_array));

	//printf("CPU time: %f seconds\n", dwalltime() - time_begin);
	//windows time
	printf("CPU time: %f seconds\n", dwalltime() - time_begin ); 
     // deallocate memory
	bool b=true;
	for(int i=0; i<size_array; i++){
		if(h_array_sec[i] !=  h_array3[i]){
			printf("GPU and CPU have different results (at least) at position %i\n", i);
			b=false;
			break;		
		}
	}
	if(b)
		printf("GPU and CPU have the same results\n");
    free(h_array3); free(h_array2); free(h_array1); free(h_array_sec);
    cudaFree(d_array3);cudaFree(d_array2);cudaFree(d_array1);
	system("pause");
	  

}

void secuential(const int a[] ,const int b[], unsigned long int c[], const int sqrt_dim){
	int dim = sqrt_dim* sqrt_dim;
	int index_i, index_j;
	//int *c= (int *)malloc ( dim * sizeof(int));
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

	
}
