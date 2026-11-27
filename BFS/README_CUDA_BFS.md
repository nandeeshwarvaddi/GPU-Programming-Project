# Parallel Breadth-First Search (BFS) Using CUDA

## 1. Introduction

This project implements **Breadth-First Search (BFS)** for an undirected
graph using CUDA C/C++. BFS is a graph traversal algorithm that visits
vertices level by level. Starting from a source vertex, it finds the
minimum number of edges needed to reach every other vertex.

In this implementation, CUDA is used to explore multiple vertices from
the same BFS level in parallel on the GPU. The program then reports the
shortest distance from the selected source vertex to every vertex in the
graph.

## 2. Graph and Input

The graph is represented using an **adjacency matrix**. The program
currently uses a static graph whose vertices and edges are defined
directly in the code.

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
edge is added in both directions in the adjacency matrix.

To test another graph, the vertex count, edge count, source vertex, and
edge list can be changed in the program.

## 3. Implementation Approach

The implementation follows a **level-by-level parallel BFS approach**.
It uses two arrays called frontiers:

-   `frontier` contains the vertices being processed at the current BFS
    level.
-   `nextFrontier` stores the newly discovered vertices that will be
    processed in the next level.

The main steps are:

1.  **Build the graph:** The CPU creates an adjacency matrix and adds
    the edges in both directions.
2.  **Initialize BFS arrays:** The `visited` array is initialized to
    zero, and the `distance` array is initialized to `-1`. The source is
    marked as visited, its distance is set to zero, and it is placed in
    the first frontier.
3.  **Allocate and copy GPU data:** `cudaMalloc()` allocates memory on
    the GPU. `cudaMemcpy()` transfers the adjacency matrix and initial
    BFS arrays from CPU memory to GPU memory.
4.  **Launch the CUDA kernel:** The `bfsKernel()` function processes
    vertices in the current frontier. Each CUDA thread takes one
    frontier vertex and checks its possible neighbors in the adjacency
    matrix.
5.  **Mark vertices safely:** When a thread finds an unvisited neighbor,
    it uses `atomicCAS()` to change that vertex's visited value from `0`
    to `1`. This ensures that only one thread successfully discovers and
    inserts a particular vertex.
6.  **Update distance and frontier:** The distance is calculated using
    `distance[v] = distance[u] + 1`. The function `atomicAdd()` reserves
    a unique position in `nextFrontier`, where the newly discovered
    vertex is stored.
7.  **Move to the next level:** After the kernel finishes,
    `cudaDeviceSynchronize()` waits for GPU execution to complete. The
    next-frontier size is copied back to the CPU, and the two frontier
    pointers are swapped. The process repeats until the frontier becomes
    empty.
8.  **Display the result:** The final distance array is copied from the
    GPU to the CPU and printed.

## 4. CUDA Concepts Used

  
  CUDA concept                        Role in this program
  ----------------------------------- -----------------------------------
  `__global__`                        Declares the `bfsKernel()` function
                                      that runs on the GPU.

  `blockIdx.x`                        Identifies the block containing a
                                      thread.

  `threadIdx.x`                       Identifies a thread within its
                                      block.

  `blockDim.x`                        Gives the number of threads in a
                                      block.

  `atomicCAS()`                       Ensures a vertex is marked as
                                      visited only once.

  `atomicAdd()`                       Assigns a unique position when
                                      adding a vertex to the next
                                      frontier.

  `cudaMalloc()`                      Allocates memory on the GPU.

  `cudaMemcpy()`                      Transfers data between CPU and GPU.

  `cudaDeviceSynchronize()`           Waits for the CUDA kernel to
                                      finish.

  `cudaFree()`                        Releases GPU memory after
                                      execution.
  

The program uses 256 threads per block. The number of blocks is
calculated from the current frontier size:

``` cpp
int blocks = (frontierSize + threads - 1) / threads;
```

This provides enough CUDA threads to process all vertices in the current
frontier.

## 5. Expected Output

For the graph defined above, with source vertex `0`, the output is:

``` text
Parallel BFS using CUDA
Number of vertices: 5
Number of edges: 5
Source vertex: 0

BFS Distances:
Vertex 0 : 0
Vertex 1 : 1
Vertex 2 : 1
Vertex 3 : 2
Vertex 4 : 2
```

The source vertex has distance `0`. Vertices `1` and `2` are one edge
away from the source. Vertices `3` and `4` are two edges away.
Therefore, all five vertices are reachable from source vertex `0`.

## 6. Conclusion

This project demonstrates how BFS can be implemented using CUDA to
explore vertices at the same level in parallel. The adjacency matrix is
used to represent the graph, while the frontier arrays organize the
traversal level by level. Atomic operations help coordinate threads when
vertices are discovered concurrently. The final distance array gives the
shortest number of edges from the source to each reachable vertex.

The current version uses a small static graph and an adjacency matrix,
which makes the implementation easy to understand. For larger sparse
graphs, a more memory-efficient representation such as Compressed Sparse
Row (CSR) could be considered.
