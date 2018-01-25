# Chapter 1

## Throughout vs. Latency oriented design
1. Throughout oriented design tries to maximize the number of threads executed in a time unit. This may mean that an individual thread runs very slowly (due to many DRAM accesses, etc), but a large number of threads will execute very quickly.
2. Latency oriented design tries to minimize the execution time for a single thread. This is the approach that CPUs take. Large caches reduce main memory lookup times, quick arithmetic units make it quick for an individual thread, etc.

## highly threaded streaming multiprocessors (SMs)

## SIMD (single instruction, multiple data)

### Message Passing Interface (MPI) for scalable cluster computing
1. Nodes in a cluster don't share memory. Data sharing must be done explicitly through message passing
2. Successful in HPC
3. The no shared memory is a hurdle for porting programs into MPI. 
4. Oftentimes, you use a CUDA / MPI mix: CUDA at the device level and MPI at the higher level.

### OpenMP for shared-memory multiprocessor systems.
1. Compiler and runtime. You use special commands and hints to where highly parallelizable parts of the code is
2. OpenACC gets to a heterogeneous model. However, The Open compilers are still evolving.

Someone once said that if you don’t care about performance, parallel programming is very easy.

## Warps

1. An SM is designed to execute all threads in a warp following the Single Instruction, Multiple Data (SIMD) model—i.e., at any instant in time, one instruc- tion is fetched and executed for all threads in the warp.
2. all threads in a warp will always have the same execution timing.
3. Latency Tolerance: Only a small subset of warps can be executed at any one time. This is how global memory accesses are masked: when one warp needs global memory, another warp can run.
4. This ability to tolerate long-latency operations is the main reason GPUs do not dedicate nearly as much chip area to cache memories and branch prediction mechanisms as do CPUs.

## compute-to-global-memory-access ratio
1. the number of  oating-point calculation performed for each access to the global memory within a region of a program.
2. In a high-end device today, the global memory bandwidth is around 1,000 GB/s, or 1 TB/s. With four bytes in each single-precision floating-point value, no more than 1000/4 = 250 giga single-precision operands per second can be expected to load. 
3. Memory bound programs: when execution speed is limited by memory access throughput.

## Memory types
1. Global memory: Read/Write accessible to host and device.
      - Long access latencies and low access bandwidths
2. Constant memory: Host can transfer to, but device can only Read
      - With appropriate access patterns, accessing constant memory is extremely fast and parallel. Currently, the total size of constant variables in an application is limited to 65,536 bytes.
3. Shared memory: Read/Write accessible only to device by block.
      - Slower than registers, but faster than global memory
4. Registers: Read/Write accessible only to device by individual thread.
      - short access latency and much higher access bandwidth when compared to global memory.
5. Automatic array variables are stored in global memory. Instead of automatic array variables, you should probably use shared memory for arrays
6. Note: There are a fixed number of registers shared amongst all threads in a block. So the more registers used, the fewer threads one can have.
