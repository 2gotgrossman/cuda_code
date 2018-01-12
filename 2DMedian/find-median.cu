#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#define xDim 32
#define yDim 32
#define blockDim 32
#define N 32

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    Point p1;
    Point p2;
    Point p3;
} Triangle;

__device__ bool inTriangle(Triangle * triangle, Point point); 
int find_median_host(Point * points, int size);
int get_max_index(int * counts, int length);
int createVector(Point * ptArr, int max, int size);

// TODO: Shared memory copies of threads

__shared__ Point pts[N*N];
__shared__ int count_for_block;
__global__ void cuda_find_median(Point * points, int * counts, int size) {
    
    int count = 0;
    int block_num = blockIdx.x;
    int thread_num = threadIdx.x + threadIdx.y * xDim;
    int threads_per_block =  xDim * yDim;
    int total_blocks = gridDim.x;
    int p1_index, p2_index, p3_index;
    Triangle tr;

    if(thread_num == 0)
        count_for_block = 0;

    if (thread_num < size ){
        int i = thread_num;
        while ( i < size){ 
            pts[i] = points[i];
            i += threads_per_block; 
        }
    }
    __syncthreads();
    assert(N*N == size);

    Point point = pts[block_num];
    int curr_thread = thread_num;

    while(block_num < size) {
        while (curr_thread < size * size) {
            p1_index = (curr_thread / size);
            p2_index = (curr_thread % size);
            if (p1_index < p2_index){
                tr.p1 = pts[p1_index];
                tr.p2 = pts[p2_index];
                for (p3_index = p2_index + 1; p3_index < size; ++p3_index) {
                    tr.p3 = pts[p3_index];
                    if (block_num != p1_index && block_num != p2_index && block_num != p3_index) {
                        if (inTriangle(& tr, point)) {
                            count += 1;

                        }
                    }
                }
            }
            curr_thread += threads_per_block;
        }
        atomicAdd(& count_for_block , count);
        if(thread_num == 0){
            counts[block_num] = count_for_block;
            count_for_block = 0;
        }
        __syncthreads();
        
        count = 0;
        block_num += total_blocks;
        point = pts[block_num];
        curr_thread = thread_num;
    }
}



int main(int argc, char * argv[]) {

    int n = N;
    int size;
    int expected_size = n*n;
    int median_index_host;
    
    cudaSetDevice(0);

    int nBytes = expected_size * sizeof(Point) ;
    Point * host_points ;
    Point * device_points ;
    int * device_counts,* host_counts;


    host_points = (Point *) malloc(nBytes) ;
    size = createVector(host_points, n, expected_size);

    host_counts = (int *) malloc(size * sizeof(int));
    for (int index = 0; index < size; index ++){
        host_counts[index] = 0;
    }

    host_points[5].x = 100.0;
    host_points[5].y = 100.0;

/*
    median_index_host = find_median_host(host_points,size) ;
    printf("\ncpu-result=\n") ;
    printf("Size: %d, MedianX: %f, MedianY: %f, Dist: %f\n", size, host_points[median_index_host].x, host_points[median_index_host].y,
           sqrt(host_points[median_index_host].x*host_points[median_index_host].x +  host_points[median_index_host].y*host_points[median_index_host].y));
*/

    cudaProfilerStart();
    // send data to cuda device
    cudaMalloc((Point **)&device_points, nBytes) ;
    cudaMalloc((int **) &device_counts, size * sizeof(int));
    cudaMemcpy(device_points, host_points, nBytes, cudaMemcpyHostToDevice) ;
    cudaMemcpy(device_counts, host_counts, size * sizeof(int), cudaMemcpyHostToDevice) ;

    dim3 grid (blockDim) ;
    dim3 block (xDim, yDim) ;
    cuda_find_median <<<grid,block, size>>> ( device_points, device_counts, size) ;

    cudaError_t err = cudaSuccess;
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(host_counts, device_counts, size * sizeof(int), cudaMemcpyDeviceToHost) ;
    
    int gpu_median_index = get_max_index(host_counts,size);
    printf("OUT: %d \n", gpu_median_index);
    printf("\ngpu-result=\n") ;
    printf("Size: %d, MedianX: %f, MedianY: %f, Dist: %f\n", size, host_points[gpu_median_index].x, host_points[gpu_median_index].y,
           sqrt(host_points[gpu_median_index].x*host_points[gpu_median_index].x +  host_points[gpu_median_index].y*host_points[gpu_median_index].y));

    cudaFree(device_points) ;
    cudaFree(device_counts) ;
    free(host_points) ;
    free(host_counts) ;

    cudaProfilerStop();
    
    return 0 ;
}

// Finds whether pt is within triangle using barycentric coordinates.
// https://stackoverflow.com/questions/13300904/determine-whether-pt-lies-inside-triangle
bool inTriangleHost(Triangle * triangle, Point pt) {
    double inv_denom = 1.0 / ((triangle->p2.y - triangle->p3.y) * (triangle->p1.x - triangle->p3.x) +
                              (triangle->p3.x - triangle->p2.x) * (triangle->p1.x - triangle->p3.y));

    double a = ((triangle->p2.y - triangle->p3.y) * (pt.x - triangle->p3.x) +
                (triangle->p3.x - triangle->p2.x)*(pt.y - triangle->p3.y)) * inv_denom;
    double b = ((triangle->p3.y - triangle->p1.y) * (pt.x - triangle->p3.x) +
                (triangle->p1.x - triangle->p3.x)*(pt.y - triangle->p3.y)) * inv_denom;
    double c = 1.0 - a - b;

    if ((a >= 0.0) && (b >= 0.0 ) && (c >= 0.0)){
        return true;
    }
    else {
        return false;
    }
}


// Finds whether pt is within triangle using barycentric coordinates.
// https://stackoverflow.com/questions/13300904/determine-whether-pt-lies-inside-triangle
__device__ 
bool inTriangle(Triangle * triangle, Point pt) {
    double inv_denom = 1.0 / ((triangle->p2.y - triangle->p3.y) * (triangle->p1.x - triangle->p3.x) +
                              (triangle->p3.x - triangle->p2.x) * (triangle->p1.x - triangle->p3.y));

    double a = ((triangle->p2.y - triangle->p3.y) * (pt.x - triangle->p3.x) +
                (triangle->p3.x - triangle->p2.x)*(pt.y - triangle->p3.y)) * inv_denom;
    double b = ((triangle->p3.y - triangle->p1.y) * (pt.x - triangle->p3.x) +
                (triangle->p1.x - triangle->p3.x)*(pt.y - triangle->p3.y)) * inv_denom;
    double c = 1.0 - a - b;

    if ((a >= 0.0) && (b >= 0.0 ) && (c >= 0.0)){
        return true;
    }
    else {
        return false;
    }
}

int find_median_host(Point * arr, int length){

    int counts[length];
    for (int l = 0; l < length; ++l) {
        counts[l] = 0;
    }


    Triangle * tr;
    tr = (Triangle *) malloc(sizeof(Triangle));



    for (int i = 0; i < length - 2; i += 1){
        tr->p1 = arr[i];
        for (int j = i+1; j < length - 1; j += 1 ){
            tr->p2 = arr[j];
            for (int k = j+1; k < length; k += 1 ){

                tr->p3 = arr[k];


                for (int index = 0; index < length; index += 1) {
                    if (index == i || index == j || index == k){
                        continue;
                    }


                    if (inTriangleHost(tr, arr[index])){
                        counts[index] += 1;

                    }

                }
            }

        }
    }
    
    int maxIndex = get_max_index(counts, length);

    free(tr);

    return maxIndex;

}

int get_max_index(int * counts, int length){
    int max = 0;
    int maxIndex = 0;

    for (int i = 0; i < length; i++){
        if (counts[i] > max){
            max = counts[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

int createVector(Point * ptArr, int max, int size){
    double rand1, rand2;
    int index = 0;
    for (int i = 0; i < max; i++){
        for (int j = 0; j < max; j++){
            if(index >= size){
                printf("ERROR!!! Index > Size");
                return -1;
            }

//            printf("Index: %d\n",index);
            rand1 = (double)rand()/RAND_MAX*2.0-1.0;  //float in range -1 to 1
            rand2 = (double)rand()/RAND_MAX*2.0-1.0;
//            printf("Rand1: %f, Rand2: %f \n", rand1, rand2);

            (ptArr+index)->x = rand1;
            (ptArr+index)->y = rand2;
            index += 1;

        }
    }

    return index ;
}
