# CPU vs GPU Parallel Breadth-First Search (BFS) Using CUDA

## 1. Introduction

This project implements Breadth-First Search (BFS) in two ways and
compares their execution time:

-   **Sequential BFS on the CPU:** visits graph vertices one by one
    using a C++ queue.
-   **Parallel BFS on the GPU:** uses a CUDA kernel to explore vertices
    in the same BFS level in parallel.

Both implementations use the same graph and source vertex. Their
distance arrays are compared at the end to check that the CPU and GPU
produce the same BFS result.

## 2. Graph and Input

The graph is represented using an **adjacency matrix**. The program uses
a static, undirected graph defined in `main()`:

``` cpp
const int V = 5;
const int E = 5;
const int source = 0;

int edges[E][2] = {
    {0, 1},
    {0, 2},
    {1, 3},
    {1, 4},
    {2, 4}
};
```

Here, `V` is the number of vertices, `E` is the number of edges, and
`source` is the starting vertex. Since the graph is undirected, every
edge is added to the adjacency matrix in both directions.

The same adjacency matrix is passed to the CPU and GPU implementations
so that both versions solve exactly the same problem.

## 3. CPU BFS Implementation

The function `cpuBFS()` implements the normal sequential BFS algorithm.

It creates a `visited` array and a C++ queue. The source vertex is
marked as visited, its distance is set to zero, and it is inserted into
the queue.

The function then repeatedly removes a vertex from the front of the
queue and checks all possible neighbors in the adjacency matrix. When an
unvisited neighbor is found, it is marked as visited, its distance is
set to one more than the current vertex's distance, and it is added to
the queue.

This continues until the queue is empty. The CPU therefore processes the
BFS traversal sequentially, one vertex at a time.

## 4. GPU Parallel BFS Implementation

The GPU version uses two parts: the CUDA kernel `bfsKernel()` and the
host function `gpuBFS()`.

### CUDA kernel

The `bfsKernel()` function performs the parallel neighbor exploration.
Each CUDA thread calculates its global thread index:

``` cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

If the index is within the current frontier size, the thread takes the
corresponding vertex and checks its neighbors in the adjacency matrix.

Two atomic operations are used:

-   **`atomicCAS()`** changes a vertex's visited value from `0` to `1`
    only if it was previously `0`. If multiple threads try to discover
    the same vertex, only one succeeds.
-   **`atomicAdd()`** increments the next-frontier counter and gives the
    successful thread a unique position in the next frontier.

When a vertex is discovered, its distance is calculated as
`distance[u] + 1`.

### GPU host function

The `gpuBFS()` function manages the GPU execution from the CPU:

1.  It initializes the visited array and the first frontier with the
    source vertex.
2.  It allocates GPU memory using `cudaMalloc()`.
3.  It copies the adjacency matrix and initial BFS data from CPU memory
    to GPU memory using `cudaMemcpy()`.
4.  It launches `bfsKernel()` with 256 threads per block. The number of
    blocks is calculated from the current frontier size.
5.  After each kernel launch, it checks the launch status and calls
    `cudaDeviceSynchronize()` to wait for the kernel to finish.
6.  It copies the next-frontier size back to the CPU, swaps the current
    and next frontier pointers, and repeats the process until no
    vertices remain in the frontier.
7.  It copies the final distance array back to the CPU and releases the
    allocated GPU memory using `cudaFree()`.

The frontier approach keeps the BFS traversal level by level. Vertices
in the same current frontier can be explored by different GPU threads at
the same time.

## 5. CPU and GPU Comparison

The `main()` function creates the graph once and prepares two separate
distance arrays: `cpu_distance` and `gpu_distance`.

The CPU version is timed using `high_resolution_clock` around the call
to `cpuBFS()`. The GPU version is timed around the call to `gpuBFS()`,
with CUDA synchronization before and after the measurement.

**Timing note:** The CPU measurement covers the CPU BFS function. The
GPU measurement includes GPU memory allocation, host-to-device data
transfers, kernel execution, device-to-host result transfer, and memory
cleanup. Therefore, the reported GPU time is an end-to-end measurement,
not only the kernel execution time.

After both versions finish, the program compares their distance arrays:

``` cpp
bool correct = (cpu_distance == gpu_distance);
```

If the arrays match, the result is displayed as `PASSED`; otherwise, it
is displayed as `FAILED`.

The program also calculates GPU speedup using:

``` cpp
double speedup = cpuTime / gpuTime;
```

A speedup greater than `1` means the measured GPU execution took less
time than the CPU execution. A value below `1` means the CPU execution
took less time for that measurement.

## 6. CUDA Functions and Concepts Used

  -----------------------------------------------------------------------
  CUDA function/concept               Purpose
  ----------------------------------- -----------------------------------
  `__global__`                        Declares the GPU kernel
                                      `bfsKernel()`.

  `blockIdx.x`, `threadIdx.x`         Identify a thread's position in the
                                      CUDA grid.

  `blockDim.x`                        Gives the number of threads in a
                                      block.

  `atomicCAS()`                       Prevents multiple threads from
                                      discovering the same vertex.

  `atomicAdd()`                       Assigns a unique position in the
                                      next frontier.

  `cudaMalloc()`                      Allocates memory on the GPU.

  `cudaMemcpy()`                      Transfers data between CPU and GPU.

  `cudaDeviceSynchronize()`           Waits for GPU work to complete.

  `cudaGetLastError()`                Checks for a CUDA kernel launch
                                      error.

  `cudaFree()`                        Releases GPU memory.
  -----------------------------------------------------------------------

The `CUDA_CHECK` macro checks CUDA API calls and prints an error message
if a call fails.

## 7. Expected BFS Result

For the five-vertex graph with source vertex `0`, both implementations
should produce:

``` text
Vertex 0 : 0
Vertex 1 : 1
Vertex 2 : 1
Vertex 3 : 2
Vertex 4 : 2
```

Vertices `1` and `2` are one edge away from the source. Vertices `3` and
`4` are two edges away. All five vertices are reachable from vertex `0`.

The program prints the CPU distances, GPU distances, their execution
times, the result verification, and the calculated GPU speedup. Actual
execution times and speedup depend on the hardware and runtime
conditions, so they should be taken from the program output rather than
assumed in advance.

## 8. Conclusion

This project compares a sequential CPU BFS implementation with a
level-by-level parallel BFS implementation using CUDA. Both versions use
the same graph and source vertex, and their distance arrays are checked
for correctness. The program also measures their execution times and
calculates the GPU speedup.

The comparison demonstrates how CUDA threads, frontiers, and atomic
operations can be used to parallelize graph traversal. Since the current
graph is small, the GPU may not be faster because GPU setup and
memory-transfer overhead can be significant. Larger graphs would provide
a more useful basis for studying performance differences.
