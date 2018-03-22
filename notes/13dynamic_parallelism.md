# Dynamic Parallelism

- Threads can create subthreads
- Programmatically, you can write a kernel launch statement within a kernel
```c
kernel_name<<< gridDim, threadDim, sharedMemSize, cudaStream >>>([kernel arguments])
```

- Classic pattern
```c
_global__ void kernel(unsigned int* start, unsigned int* end,float* someData, float* moreData) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  doSomeWork(someData[i]);

  for(unsigned int j = start[i]; j < end[i]; ++j) {
    doMoreWork(moreData[j]);
  }
}
```

- Using Dynamic Parallelism
```c
__global__ void kernel_parent(unsigned int* start, unsigned int* end, float* someData, float* moreData) {
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; doSomeWork(someData[i]);
  kernel_child <<< ceil((end[i]-start[i])/256.0) , 256 >>> (start[i], end[i], moreData);
}

__global__ void kernel_child(unsigned int start, unsigned int end, float* moreData) {
  unsigned int j = start + blockIdx.x*blockDim.x + threadIdx.x;
  if(j < end) { 
    doMoreWork(moreData[j]);
  } 
}
```

## Memory Rules
1. Global Memory
    - A parent and child thread can make global memory accessible to each other
    - Weak consistency(?)
    - Consistent when (1) child grid is created and (2) when the child grid completes
        - Parent thread and determine completion by syncronizing
2. Constant memory
    - Kernels cannot write to constant memory
3. Local memory 
    - Local memory is not visible visible to children threads
    - Declaring `__device__` memory outside of a kernel will be globally accessible
    - There are warnings
4. Shared memory
    - Children do not have access
5. In summary, use global memory (`malloc`) or `__device__` memory for sharing data between a parent and child

### `malloc`ing meory
1. If you are a device or host, you can only `cudaFree` memory that has been `cudaMalloc`ed from a device or host, respectively
2. `cudaLimitMallocHeapSize` determines an allocation limit for device-allocated memory

## Synchronization
1. Kernel launches from device are nonblocking
2. If synchronization is required, need to explicitly call `cudaDeviceSynchronize()`
    - Parent kernel will wait until all children threads have synchronized
3. If a block of parent threads must be synchronized, then a call to `__syncthreads()` must follow
4. If not implicity, children will be synchronized before the parent kernel terminates

## Max Levels of Depth
1. About ~150 MB of memory is needed per level of synchronization depth
2. This is because when waiting for children threads to complete, the parent might be swapped out of execution
    - Need to store state
3. Default memory reserved for backing store is enough for two levels
    - Can increase by calling `cudaDeviceSetLimit()`
4. The number of grids is by default 2048. If you need more grids, they will be virtualized (swapped out of execution). Obviously hurts performance
5. However, you can and are recommended to set the device limit yourself to the number of grids launched
    - Oftentimes, you will launch as many grids as the number of parent threads
6. Since there are performance losses the more levels you divide, it is often a good idea to build into your code a max depth
    - For example: Only subdivide in merge sort a certain number of times


# Concepts left to explore in CUDA
1. Streams and Events
