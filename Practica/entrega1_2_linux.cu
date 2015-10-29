
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

float secuential(const int array[] , int dim){
	float mean=0;
	for(int i=0; i<dim;i++){
		mean+=array[i];
	}
	mean=mean/dim;
	float sum=0;
	for(int i=0; i<dim;i++){
		sum+=(array[i]-mean)*(array[i]-mean);
	}
	return sqrt(sum/(dim-1));

}

     
__global__ void func( const int array[] , int dim,float result[], const int thread_number)
{
    int index = blockIdx.x* blockDim.x* blockDim.y* blockDim.z+threadIdx.z* blockDim.y* blockDim.x+ threadIdx.y* blockDim.x+ threadIdx.x;
 	//printf("sum:%i\n",  result[0]);
	if(index<dim){
		if(dim<=thread_number){ //if more threads than array size
		//	printf("Thread %i; Adding value of index %i\n", index, index, array[index]);
			atomicAdd(result,array[index]);
		}
		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
			//		printf("Thread %i; Adding value of index %i\n", index, i, array[i]);
					atomicAdd(result,array[i]);
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim/thread_number); i< dim; i++){
				//	printf("Thread %i; Adding value of index %i\n",index, i, array[i]);
					atomicAdd(result,array[i]);
				}
			}
		}
		//printf("sum:%i\n",  result[0]);
	}
	
	 __syncthreads();

	float mean=result[0]/dim;
	result[0]=0;
	if(index<dim){
		if(dim<=thread_number){ //if more threads than array size
			//printf("Thread %i; Adding value of index %i\n", index, index, array[index]);
			atomicAdd(result,(array[index]-mean)*(array[index]-mean));
		}
		else{ //if less threads than array size
			if(index!=thread_number-1){//if not last thread deal with size_array/thread_nb array entries
				for(int i=index*(int)(dim/thread_number); i< index*(int)(dim/thread_number)+(int)(dim/thread_number); i++){
					//printf("Thread %i; Adding value of index %i\n", index, i, array[i]);
					atomicAdd(result,(array[i]-mean)*(array[i]-mean));
				}
			}
			else{ //if last thread deal with all remaining array entries
				for(int i=index*(int)(dim/thread_number); i< dim; i++){
					//printf("Thread %i; Adding value of index %i\n",index, i, array[i]);
					atomicAdd(result,(array[i]-mean)*(array[i]-mean));
				}
			}
		}
	}
	 __syncthreads();
	result[0]=sqrt(result[0]/(dim-1));
} 


    
int main(int argc, char *argv[]){
	//Measure time
	clock_t time_begin;
	
    // pointers to host & device arrays
     int *device_array = 0;
     int *host_array = 0;
	 int size_array=9;
	 float *d_gpu_res=NULL;
	 float *h_gpu_res= 0;
	 bool verbose=false;
	 int N=1;
	 if(argc == 3){
		 size_array=atoi(argv[1]);
		 N=atoi(argv[2]);
	 }
	 else if(argc== 4 ){
		 size_array=atoi(argv[1]);
		 N=atoi(argv[2]);
		 verbose=(argv[3][0]=='v');
	 }
	 
	 h_gpu_res=( float*)malloc(sizeof( float));
	 h_gpu_res[0]=0;
      // malloc a host array
     host_array = (int*)malloc( size_array * sizeof(int));
	
    for(int i=0; i<size_array; i++){
		host_array[i]=rand()%10;
		if(argc==4 && verbose){
			printf("%i\t", host_array[i]);
		}
		else if(argc==4)
			host_array[i]=atoi(argv[3]);
    }
    printf("\n");
	
	
     // cudaMalloc a device array
     cudaMalloc(&device_array,size_array * sizeof(int));    
	 cudaError_t er=cudaMalloc(&d_gpu_res, sizeof(float));  
    // download and inspect the result on the host:
    cudaError_t e=cudaMemcpy(device_array, host_array, sizeof(int)*size_array, cudaMemcpyHostToDevice); 
	cudaError_t error=cudaMemcpy(d_gpu_res, h_gpu_res, sizeof(int), cudaMemcpyHostToDevice);
	//cudaerrorinvalidvalue(11)

    dim3 bloque(N,N); //Bloque bidimensional de N*N hilos
    dim3 grid(1,1);  //Grid bidimensional de M*M bloques
	int thread_number= N*N;
	if (N*N > 512){
            bloque.x = 512;
            bloque.y = 512;
            grid.x = ceil(double(N)/double(bloque.x));
            grid.y = ceil(double(N)/double(bloque.y));
     }
	time_begin=dwalltime();
    func<<<grid, bloque>>>(device_array, size_array , d_gpu_res, thread_number);
    cudaThreadSynchronize();
    // download and inspect the result on the host:
   //cudaMemcpy(host_array, device_array, sizeof(int)*size_array, cudaMemcpyDeviceToHost); 
	cudaMemcpy(h_gpu_res, d_gpu_res, sizeof(int), cudaMemcpyDeviceToHost); 
	printf("GPU time: %f seconds\t", dwalltime() - time_begin );
	printf("GPU result: %f\n", h_gpu_res[0]);
	
	time_begin=dwalltime();
	float cpu_res=secuential(host_array, size_array);
	printf("CPU time: %f seconds\t", dwalltime() - time_begin  ); 
	printf("CPU result: %f\n", cpu_res);

     // deallocate memory
      free(host_array);free(h_gpu_res);
      cudaFree(device_array); cudaFree(d_gpu_res);

	

}