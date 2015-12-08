#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void secuential(const int a[] ,const int b[], int c[], const unsigned int sqrt_dim);
     
__global__ void multiply(  const int* A, const int* B,int* C, int width, int tile_width)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

	
    // Index of the first sub-matrix of A processed by the block
    int aBegin = width * tile_width * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + width - 1;
		
    // Step size used to iterate through the sub-matrices of A
    int aStep  = tile_width;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = tile_width * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = tile_width * width;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;
	

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

		extern __shared__ int shared[];

		int *As=&shared[0];
		int *Bs=&shared[tile_width*tile_width];

        As[ty+tile_width*tx] = A[a + width * ty + tx];
        Bs[ty+tile_width*tx] = B[b + width * ty + tx];
        __syncthreads();

		//Csub=B[b + width * ty + tx] + A[a +  ty + width*tx];
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < tile_width; ++k)
        {
            //Csub += As[ty+tile_width*k] * Bs[k+tile_width*tx];//a*b
			Csub += As[ty+tile_width*k] * Bs[tx+tile_width*k]; //a*b^t
        }
		
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = width * tile_width * by + tile_width * bx;
    C[c + width * ty + tx] = Csub;
	C[c + width * ty + tx]+=B[c + width * ty + tx] + A[c + width * tx + ty];

}

void constantInit(int *data, int size, int val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }

}

void randomInit(int* data, int size)
{
   for (int i = 0; i < size; ++i)
   data[i] = rand()%10;//rand() / (float)RAND_MAX;
}

 
int main(int argc, char** argv)
{
	clock_t time_begin;
	unsigned int size_array  = (argc > 1)? atoi (argv[1]): 1024;
	unsigned int tile_width = (argc > 2)? atoi (argv[2]): 2;	
	bool verbose= (argc>3)? (argv[3][0]=='v'): false;

   int* h_array1 = (int*) malloc(sizeof(int) * size_array*size_array); 
   int* h_array2 = (int*) malloc(sizeof(int) * size_array*size_array);
   int* h_array3 = (int*) malloc(sizeof(int) * size_array*size_array);
   int* h_array_sec = (int*) malloc(sizeof(int) * size_array*size_array);
 

   constantInit(h_array1, size_array*size_array,1);
   constantInit(h_array2, size_array*size_array,1);
    
	if(verbose){
		printf("A:\n");
		for(int i=0; i<size_array*size_array; i++){
			printf("%i\t", h_array1[i]); 
			if((i+1)%size_array==0) printf("\n");
		}
		printf("\n");
		printf("B:\n");
		for(int i=0; i<size_array*size_array; i++){
			printf("%i\t", h_array2[i]);
			if((i+1)%size_array==0) printf("\n");
		}
		printf("\n");
	}
  
   int *d_array1,*d_array2, *d_array3;
   cudaMalloc((void**) &d_array1, sizeof(int) * size_array*size_array);
   cudaMalloc((void**) &d_array2, sizeof(int) * size_array*size_array);
   cudaMalloc((void**) &d_array3, sizeof(int) * size_array*size_array);
   cudaMemcpy(d_array1, h_array1, sizeof(int) * size_array*size_array, cudaMemcpyHostToDevice);
   cudaMemcpy(d_array2, h_array2, sizeof(int) * size_array*size_array, cudaMemcpyHostToDevice);

   dim3 bloque(tile_width, tile_width);
   dim3 grid(size_array / bloque.x, size_array / bloque.y);
   time_begin=clock();
   multiply<<< grid, bloque, tile_width*tile_width*tile_width*tile_width >>>( d_array1, d_array2,d_array3, size_array, tile_width);
   cudaMemcpy(h_array3, d_array3, sizeof(int) * size_array*size_array, cudaMemcpyDeviceToHost);
   printf("GPU time: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); 

   if(verbose){
		printf("Array C=B + AB^t + A^t :\n");
		for(int i=0; i<size_array*size_array; i++){
			printf("%i\t", h_array3[i]);
			if((i+1)%size_array==0) printf("\n");
		}
	}
	time_begin=clock();
	secuential(h_array1, h_array2, h_array_sec, size_array);
	printf("CPU time: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); 

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
   free(h_array1);
   free(h_array2);
   free(h_array3);
   cudaFree(d_array1);
   cudaFree(d_array2);
   cudaFree(d_array3);
 
}

void secuential(const int a[] ,const int b[], int c[], const unsigned int sqrt_dim){
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
