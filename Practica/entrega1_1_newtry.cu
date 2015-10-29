#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


//M and N number of threads (grid and block)

void secuential(const int a[] ,const int b[], int c[], const int sqrt_dim);
     
__global__ void multiply( const int a[] ,const int b[], int c[] , const int sqrt_dim, const int thread_number)
{
  
	unsigned int ROW = blockIdx.y*blockDim.y+threadIdx.y;  //index_j?
    unsigned int COL = blockIdx.x*blockDim.x+threadIdx.x; //index_i?

    float tmpSum = 0;
	if(thread_number >= sqrt_dim * sqrt_dim){
		if (ROW < sqrt_dim && COL < sqrt_dim) {
			// each thread computes one element of the block sub-matrix
			for (int i = 0; i < sqrt_dim; i++) {
				tmpSum += a[ROW *sqrt_dim + i] * b[i * sqrt_dim + COL];
			}
		}
	
		c[ROW * sqrt_dim + COL]= b[ROW * sqrt_dim + COL]; //c= b
		c[ROW * sqrt_dim + COL]+= a[COL + ROW * sqrt_dim]; //c+= a^t
		c[ROW * sqrt_dim + COL]+= tmpSum;
	}
	else{
		unsigned int index=ROW * sqrt_dim + COL;
		unsigned int dim=sqrt_dim*sqrt_dim;
		
		if(index!=(thread_number-1)){//if not last thread deal with size_array/thread_nb array entries
			//for(int j=index*(int)(dim/thread_number); j< index*(int)(dim/thread_number)+(int)(dim/thread_number); j++){
				
				for(unsigned int r=ROW; r<= ROW + (int)(sqrt_dim/thread_number); r++){
					for(unsigned int cl=COL; cl <= COL +(int)(sqrt_dim/thread_number); cl++){

						tmpSum=0;
						if (r < sqrt_dim && cl < sqrt_dim) {
							// each thread computes one element of the block sub-matrix
							for (int i = 0; i < sqrt_dim; i++) {
								tmpSum += a[r *sqrt_dim + i] * b[i * sqrt_dim + cl];
							}
						}
	
						c[r * sqrt_dim + cl]= b[r * sqrt_dim + cl]; //c= b
						c[r * sqrt_dim + cl]+= a[COL + r * sqrt_dim]; //c+= a^t
						c[r * sqrt_dim + cl]+= tmpSum;
					} 
				}
			//}
			
				
		}
		
		else{ //if last thread deal with all remaining array entries
	
			for(unsigned int r=ROW; r<sqrt_dim; r++){
				for(unsigned int cl=COL; cl<sqrt_dim; cl++){
					tmpSum=0;
					if (r < sqrt_dim && cl < sqrt_dim) {
						// each thread computes one element of the block sub-matrix
						for (int i = 0; i < sqrt_dim; i++) {
							tmpSum += a[r *sqrt_dim + i] * b[i * sqrt_dim + cl];
						}
					}
	
					c[r * sqrt_dim + cl]= b[r * sqrt_dim + cl]; //c= b
					c[r * sqrt_dim + cl]+= a[COL + r * sqrt_dim]; //c+= a^t
					c[r * sqrt_dim + cl]+= tmpSum;
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
	  int *h_array_sec= 0;
	  int size_array=16; //here, size_array =L has to be a square

	 int N=3;
	 if(argc == 3){
		 size_array=atoi(argv[1]) * atoi(argv[1]) ;
		 N=atoi(argv[2]);
	 }  
      // malloc columns of host arrays
     h_array1 = (int*)malloc( size_array * sizeof(int));
	  h_array_sec= (int*)malloc( size_array * sizeof(int));
	  h_array2 = (int*)malloc( size_array * sizeof(int));
	  h_array3 = (int*)malloc( size_array * sizeof(int));
	
	  
	//printf("Array A:\n");
	for(int i=0; i<size_array; i++){
		h_array1[i]=1;//rand()%10;
	//	printf("%i\t",  h_array1[i]);
		//if((i+1)%(int)sqrt((float)size_array)==0)
		//	printf("\n");
	}
	//printf("\n");
 
	//printf("Array B:\n");
	for(int i=0; i<size_array; i++){
		h_array2[i]=1;//rand()%10; 	
		//printf("%i\t",  h_array2[i]);
		//if((i+1)%(int)sqrt((float)size_array)==0)
		//	printf("\n");
	}
	//printf("\n");
 

     // cudaMalloc a device array
    cudaMalloc(&d_array1,size_array * sizeof(int));    
	cudaMalloc(&d_array2,size_array * sizeof(int));  
	cudaMalloc(&d_array3,size_array * sizeof(int));  
    // download and inspect the result on the host:
    cudaMemcpy(d_array1, h_array1, sizeof(int)*size_array, cudaMemcpyHostToDevice);   
	cudaMemcpy(d_array2, h_array2, sizeof(int)*size_array, cudaMemcpyHostToDevice);   

   // dim3 bloque(N,N); //Bloque bidimensional de N*N hilos (max 512 threads in a block)
    //dim3 grid(M,M);  //Grid bidimensional de M*M bloques
	int thread_number= N*N;
	printf("%i threads, %ix%i matrix\n", thread_number,  (int)sqrt((float)size_array), (int)sqrt((float)size_array));
	time_begin=clock();

	dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        }
    multiply<<<threadsPerBlock, blocksPerGrid>>>(d_array1, d_array2 , d_array3,sqrt((float)size_array), thread_number);
	 cudaDeviceSynchronize();
	//cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(h_array3, d_array3, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
	
	//printf("GPU time: %f seconds\n", clock() - time_begin);
	//windows time
	printf("GPU time, %i threads: %f seconds\n", thread_number,(((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s

	printf("Array C=B + AB^t + A^t :\n");
    for(int i=0; i<size_array; i++){
        printf("%i\t", h_array3[i]);
	if((i+1)%(int)(sqrt((float)size_array))==0)
		printf("\n");
	}
	printf("\n");
	time_begin=clock();
	secuential(h_array1, h_array2,  h_array_sec, sqrt((float)size_array));

	//printf("CPU time: %f seconds\n", clock() - time_begin);
	//windows time
	printf("CPU time: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s
     // deallocate memory
	for(int i=0; i<size_array; i++){
		if(h_array_sec[i] !=  h_array3[i]){
			printf("GPU and CPU have different results at position %i\n", i);
			break;		
		}
	}
    free(h_array3); free(h_array2); free(h_array1); free(h_array_sec);
    cudaFree(d_array3);cudaFree(d_array2);cudaFree(d_array1);

	  

}

void secuential(const int a[] ,const int b[], int c[], const int sqrt_dim){
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
	//free(c);
}
