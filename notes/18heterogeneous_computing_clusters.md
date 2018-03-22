# Heterogeneous computing clusters
1. Top supercomputers use a mix of GPUs and CPUs

# MPI: Message Passing Interface
1. Set of API functions for communicating between processes running in a computing cluster.
2. Distributed memory model where processes exchange information by sending messages to each other
3. Abstracts away the network connection
4. Can address each other using unique IDs
5. SPMD model

### MPI Basics
1. `MPI_Init()`
2. `MPI_Comm_rank()` - Creates unique ID for each process
    - Called MPI rank, or PID
    - `0` to `nProcesses - 1`
3. `MPI_Comm_size()` - returns number of processes
4. `MPI_Finalize()`

### Sending and Receiving Messages
```c
MPI_Send(void *startingAddress, 
         int numElemsToSend, 
         MPI_Datatype datatype, 
         int destination,
         int messageTag,
         MPI_Comm communicatorHandle)

MPI_Recv(void *startingAddress, 
         int numElemsToSend, 
         MPI_Datatype datatype, 
         int destination,
         int messageTag,
         MPI_Comm communicatorHandle)
```
- Blocking Communication
- `MPI_Isend()` and `MPI_Irecv()` are nonblocking
    - `MPI_wait()` and `MPI_Test()` allows you to see if communication has finished
- There is an `MPI_ANY_TAG`: Receiver will receive from anyone

### MPI I/O
1. I/O have a lot of system complexity

## Running Example
1. Heat transfer using Jacobi Iterative Method
    - Take a volume and break it into cubes
    - Heat transfer is calculated in discrete steps
    - New state of a cube depends on the last state of its 6 neighbors that share a face with it

### First idea of solution
- Each node gets a cube
    - Each node will have that cube subdivided into smaller cubes
    - We can run computations of this smaller cube on a CUDA device
- The Process
    1. Send and Receive temps for step N
    2. Calculate temp for stage N+1
    3. Repeat
- Issues
    - In part 1, we flood the network with data and have nothing to compute
    - In part 2, we have an empty network and have too much to compute

### Improved solution
- Modify solution 1 as follows
    1. First, calculate the boundary slices of Step N that neighboring nodes will need for Step N+1
        - We can prioritize the boundary slices in CUDA memory by using _pinned memory allocation_
            - AKA page locked memory
            - The memory allocation will not be paged out of the operating system
            - Allows GPU to directly access memory from RAM without involving the CPU
    2. Communicate and Complete Subcubes Concurrently
        a. Communicate (send and receive) the data to neighboring nodes for Step N+1
        b. Compute each subcube for Step N
        - Non-blocking until we get to Step 3
    3. Block and Repeat
- We can use `cudaStreams` to run steps (1 and 2a) and step b concurrently

## MPI Collective Communication
- Tasks that involve a group of nodes
- Most common: `MPI_Barrier()`
- They are highly optimized. Use them over creating combinations of send and receive calls

## CUDA-Aware MPI
1. Sending and receiving messages from GPU memory
2. Removes need of device-to-host data transfers before sending MPI messages
3. Removes need to use host-pinned memory buffers

# OpenACC
- Developed by Cray Enterprises, CUDA, and some rando
    - Support of multiple universities and national laboratories
- Has gained a lot of support from vendors, comapnies, etc
- A specification of compiler directives and API routines for writing data parallel code in C, C++, and Fortran
    - Can be compiled to parallel architectures on GPUs and multicore CPUs
- Programmer annotates loops and data structures that the OpenACC compiler will target
- Goal
    - Provide a programming model for domain experts
    - Maintain single source code for multiple architectures
    - Portable Performance: If performs well on one architecture, then it performs well on others

## OpenACC and CUDA
- OpenACC compiler will
    - Generate kernels
    - Create the register and shared memory variables
    - Apply some performance optimizations
- Good interface for skilled CUDA programmers to quickly parallelize large applications

## OpenACC Execution Model
- Begins on a _host_ CPU
- Host will offload execution to an _accelerator device_
    - The accelerator may be the same device as the host
    - For example, a GPU connected to a CPU via PCIe
    - Allows for host and device to have physically separate memories or shared memory
    - Most portable way: Assume distinct accelerator with distinct memory

### Syncronicity and Data Sharing
- By default, enforces syncronous behavoir
    - At the end of eah parallel execution, the host and device sync by default
    - Similar to `fork()` and `join()` of POSIX-threads
- By default, OpenACC treats data as if there is always one copy that lives in either host or device
    - Modification in one place will be reflected in the other at some point in time
    - Means for controlling how data is moved between host and device and shared between different nodes
    - Can't assume shared memory

### Levels of Data Parallism
1. Gangs
    - Fully independent execution units
    - Cannot sync amongst gangs nor exchange data
    - No assumption about order of execution
    - Similar to CUDA thread blocks
2. Workers
    - One or more workers per gang
    - Shared cache memory
    - Can be synced
    - Similar to CUDA threads
3. Vectors
    - A vector is an operation that is computed on multiple data elements in the same instruction
    - Workers operate on vectors
- Can have sequential loops at any level



## OpenACC Directive Format
```c
#pragma acc <directive> <clauses>
```
- `acc` is the prefix for OpenACC directives
- Incremental path for moving existing code to accelerators
- Code will run correctly if directives are ignored
- When accelerating, compiler will in general copy any variable on the RHS of an `=` operator to the accelerator and copy any variable on the LHS of an `=` operator back to the host

### Example directives
1. `kernels` directive
    - `#pragma acc kernels`
    - Tells compiler that the region that follows contains loops that should be transformed into one or more accelerator kernels
        - The _desire_ to parallelize the following code
    - Only will run if the compiler deems the loops to be safely parallelizable
2. `parallel` directive
    - The command to parallelize the following code
    - If the `loop` directive is included, it will make assertions about the feasability of the loop to be accelerated without the compiler doing detailed analysis
    - Added to each loop nest to be accelerated
3. `collapse` clause
    - The outer loop is free of data races and available to parallelize, but the inner loop should be parallelizable too
    - `#pragma acc parallel loop collapse(2)`
4. `reduction`  clause
    - Basically a `reduce` in map-reduce

### Data Directives
- `create`: `malloc` space on the accelerator device at the beginning of the accelerated region and `free` at the end
- `copyin`: `malloc` the variables on the accelerator device and then copy values from host
- `copyout`: `malloc` on accelerator device and copy back to host
- `copy`: `copyin` and `copyout`
- `present`: Assume the variable already exists and no need to `malloc`, copy, or `free`
- `update`: Copies memory either from host to accelerator or from accelerator to host without `malloc`ing or `free`ing

### Optimization Directives
1. `gang`, `worker`, `vector`, and `seq` inform the compiler which level of parallelism should be applied
2. Specifying `num_workers`, `num_gangs`, etc
3. Can provide a `device_type`- For CUDA, it is `nvidia`
4. `tile` directive to take advantage of data locality

### `routine` Directive
1. Function calls within parallelizable regions are difficult for a compiler
2. `routine` directive reserves specified levels of parallelism for loops within a function

## Asynchronicity
- Defaulting to synchronous computations ensures correctness
- You can opt into asynchronous behavior using the `async` clause
- CPU can either enqueue more work for the accelerator by putting the work in a asynchronous work queue
- `wait`: synchronize
- Can also have asynchronicity on accelerator device level


## CUDA and OpenACC Interoperability
1. You can call CUDA libraries with OpenACC arrays
2. CUDA Pointers can be used in OpenACC
3. CUDA device kernels can be called from OpenACC
    - Best to implement a `seq` routine

## Problems with OpenACC
1. Difficulty dealing with complex data structures
    - C++ Classes
    - C `struct`s containing pointers
2. Deep copy will hopefully be coming in version 3.0 (currently on version 2.6)
    - Will need to be accompanied by more complex memory hierarchies

