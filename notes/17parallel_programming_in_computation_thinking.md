# Parallel programming and computational thinking

1. We will generalize parallel programming to a method how how to decompose a problem into subproblems that we can solve with well-known, efficient, algorithms and numerical methods
2. Transforming problems into ones we already know how to solve
    - Need problem decomposition: Which parts are inherently serial, which are parallel, etc

## Main goals of parallel computering
1. Solve a problem in less time - Decrease latency
2. Solve bigger problems within a fixed period of time - Increase throughput
3. Achieve better solutions for a given problem and a given amount of time
    - Decrease latency _and_ increase throughput

## Good candidtates for parallel computing
1. Large problem size
    - Lots of data
2. High modeling complexity
    - Expensive operation for each datum or group of data
    - Many iterations need to be performed on the data
3. Still limited by serial parts of application (atomic operations included)

## Problem Decomposition
- Easy in theory, hard in practice
- You may come up with two difference parallel solutions, but the efficiency may vary greatly
    - Rules of thumb to reason about differences
    - Especially difficult for large problems
- Possible to mix data-level and task-level parallelism
    - Mixing CUDA and MPI

# Algorithms
Must have 3 properties:
1. Definiteness
    - Each step is precisely state
    - No room for ambiguity
2. Effective computability
    - Can be carried out by a computer
3. Finiteness
    - Guarantee of termination

## Computational Thinking
1. Need to have a good idea of the system you are working with

## Space-Time Trade-offs
1. Data locality in time of access
    - 
2. data locality in access patterns
    - 
3. Data sharing
    - Excessive data sharing reduces parallel execution
    - Don't want data throughput to be a bottleneck
    - Want to keep localized. Can improve memory bandwith efficiency without creating conflicts
    - Can be achieved through syncronization of tasks and coordinating data accesses
        - Time tradeoff: Have to wait for sync-up, but memory access could be quick if coalesced access

## SPMD: Single Program Multiple Data - The CUDA Grid Model
1. SIMD: Single _instruction_ Multiple Data
    - Special case of SPMD: threads move in lock-step
2. Affects algorithm structures and coding styles

### Typical Structure of SPMD Programs
1. Initialize - create data structures and communication chanels
2. Uniquify - Give each thread a unique ID 
3. Distribute Data - Decompose global data into chunks and localize data
4. Compute - run the computation
5. Finalize - Recreate global data structure and prepare for next step or iteration

## Strategies for Computational Thinking
1. Tune core software for hardware architectures
    - Accelerating legacy program code
    - Easiest and only small reward
2. Innovate at the algorithm level
    - Using new parallelism techniques and being clever about tradeoffs
3. Restructure the mathematical formulation
 

