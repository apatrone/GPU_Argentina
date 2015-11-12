#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>



//M and N number of threads (grid and block)

void secuential(const int a[] ,const int b[], unsigned long int c[], const int sqrt_dim);
     
__global__ void multiply( const int a[] ,const int b[], unsigned long int c[] , const int width,const int tile_width)
{

 int sum = 0;
 int col = blockIdx.x*tile_width + threadIdx.x;
 int fila = blockIdx.y*tile_width + threadIdx.y;
 if(col < width && fila < width) {
	for (int k = 0; k < width; k++)
		sum += a[fila * width + k] * b[col * width + k]; // a*b^t 

	c[fila * width + col] = sum; 
	c[fila * width + col]+= b[fila * width + col]; //c+=b
	c[fila * width + col]+= a[col*width+fila]; // c+=a^t??

 }
	
} 

    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
    // pointers to host & device arrays
      int *d_array1 = 0,*d_array2 = 0; unsigned long int *d_array3 = 0;
      int *h_array1 = 0,*h_array2 = 0;unsigned long int*h_array3 = 0;
	  unsigned long int *h_array_sec= 0;
	 unsigned int size_array=2048*2048; //here, size_array =L has to be a square
	 bool verbose=false;
	 int tile_width =16;
	 if(argc == 3){
		 size_array=atoi(argv[1]) * atoi(argv[1]) ;
		 tile_width=atoi(argv[2]);
	
	 }
	 else if(argc==4){
		 size_array=atoi(argv[1]) * atoi(argv[1]) ;
		 tile_width=atoi(argv[2]);
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
	cudaMalloc(&d_array3,size_array * sizeof(unsigned long int));  
    // download and inspect the result on the host:
    cudaMemcpy(d_array1, h_array1, sizeof(int)*size_array, cudaMemcpyHostToDevice);   
	cudaMemcpy(d_array2, h_array2, sizeof(int)*size_array, cudaMemcpyHostToDevice);   

	dim3 bloque(tile_width, tile_width);
	dim3 grid((int)ceil(double(sqrt((float)size_array))/double(bloque.x)), ceil(double(sqrt((float)size_array))/double(bloque.y)));
	//int thread_number= N*N;
	printf("%i threads per block, %ix%i matrix\n", tile_width*tile_width,  (int)sqrt((float)size_array), (int)sqrt((float)size_array));
	time_begin=clock();
    
	multiply<<<grid, bloque>>>(d_array1, d_array2 , d_array3,sqrt((float)size_array), tile_width);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
    cudaMemcpy(h_array3, d_array3, sizeof(unsigned long int)*size_array, cudaMemcpyDeviceToHost); 
	
	//printf("GPU time: %f seconds\n", clock() - time_begin);
	//windows time
	printf("GPU time, %i threads per block: %f seconds\n", tile_width*tile_width,(((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s


	if(verbose){
		printf("Array C=B + AB^t + A^t :\n");
		for(int i=0; i<size_array; i++){
			printf("%i\t", h_array3[i]);
		if((i+1)%(int)(sqrt((float)size_array))==0)
			printf("\n");
		}
		printf("\n");
	}
	time_begin=clock();
	secuential(h_array1, h_array2,  h_array_sec, sqrt((float)size_array));

	//printf("CPU time: %f seconds\n", clock() - time_begin);
	//windows time
	printf("CPU time: %f seconds\n", (((float)clock() - (float)time_begin) / 1000000.0F ) * 1000  ); //1.18s
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
	//free(c);
}
