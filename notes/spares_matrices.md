# Sparse Matrices

## Storage
1. Dictionary of keys
    - Key is a tuple of indices (for multidimensional arrays), value is the datum
    - Good for creating the sparse matrix in random order
    - Bad for iterating over non-zero values
    - Good intermediate step between full matrix and a more efficient format
2. List of Lists
    - One list per row, entries contain column index and value
    - Typically kept sorted
    - Good for incremental matrix construction
3. Coordinate list
    - A single list of `(row, column, value)` tuples
    - Good for incremental construction
4. Compressed Sparse Row
    - Basically holds a prefix summed representation of the 2D array
5. Compressed Sparse Column
    - Column-based instead of row-based

## Special Cases

### Banded Matrix
- The values are mostly along the diagonal
- Bandwith is measure by the largest distance a non-zero lies from the origin
- There are simplified algorithms for matrices with reasonably small bandwiths

### Diagonal Matrix
- Just store the diagonal entries

### Symmetric Matrices
- Arises when you have the adjacency matrix of an undirected graph
- Just store the adjacency list

[comment]: TODO
## Cholesky Algorithm

## Tim Davis Talk
- Single-core, multi-core, single-GPU, multi-GPU
- Stitching images together
    - Sparse bundle adjustment problem
        - Non-linear least squares problem
        - Linear Least Squares Problem
- Collects matrices to test his libraries
    - Need problems with real structure
    - Can't use random data
- Can't assume RAM model
    - Really is an O(N) operation where N is the number of elements in RAM / size of RAM
    - A cubic matrix with side length N will have "random" access time on the order of O(N^3)

### Using Graph Theory to Solve Sparse Matrix Problems
= Lower Triangular Solve
    - For `Lx = b` where `L` is lower triangular, `L, x, b` are sparse, you can think of `L` as a graph of edges and do a topological sort on it.
    - Since it is directed and lower-triangular, you know there is a topological sort. Definitely no cycles
- Wrote MATLAB's sparse matrix package
- [](http://faculty.cse.tamu.edu/davis/research_files/Stanford2013.pdf)
- [](http://research.nvidia.com/sites/default/files/pubs/2011-06_Parallel-Solution-of/nvr-2011-001.pdf)

### LU Factorization
- Factoring a matrix into a lower triangular matrix and an upper triangular matrix

### Optimize around cliques
- Cliques are dense submatrices

