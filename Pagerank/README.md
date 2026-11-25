# PageRank – Parallel Implementation Using CUDA

## 1. Overview

PageRank is a graph algorithm used to calculate the importance of vertices based on the incoming links from other vertices.

This project implements the PageRank algorithm using NVIDIA CUDA to utilize GPU parallel processing and improve computational performance compared to the sequential CPU implementation.

## 2. Objective

- Implement the PageRank algorithm using CUDA.
- Parallelize PageRank computation using GPU threads.
- Compare sequential CPU and parallel GPU implementations.
- Analyze GPU performance using different thread block configurations.
- Verify the correctness of GPU results against CPU results.

## 3. Technologies Used

- C++
- NVIDIA CUDA
- NVCC Compiler
- Compressed Sparse Row (CSR) Graph Representation
- Git and GitHub

## 4. Algorithm Details

The PageRank algorithm calculates the importance of each vertex iteratively.

The formula used is:

PR_new(v) = (1-d)/N + d * (danglingSum/N + Σ(PR_old(u)/outdeg(u)))

Where:

- N = Number of vertices.
- d = Damping factor (0.85).
- outdeg(u) = Outgoing degree of vertex u.
- danglingSum = Total PageRank of vertices without outgoing edges.

## 5. GPU Implementation

The implementation uses the following CUDA kernels:

1. **init_kernel:** Initializes the PageRank value of every vertex to 1/N.
2. **contrib_kernel:** Calculates the contribution of each vertex and computes the dangling rank sum using shared memory reduction.
3. **update_kernel:** Computes new PageRank values in parallel and calculates the convergence error.

Each CUDA thread processes one vertex.

Two PageRank buffers are used to maintain synchronous iterations. Atomic operations accumulate dangling rank and convergence error.

## 6. Implementation Workflow

1. Read the graph from an edge-list file or generate a random graph.
2. Construct the incoming-edge CSR representation.
3. Execute the sequential CPU PageRank implementation.
4. Allocate GPU memory and transfer graph data.
5. Initialize PageRank values on the GPU.
6. Calculate vertex contributions.
7. Update PageRank values in parallel.
8. Check convergence using L1 error.
9. Transfer GPU results to the CPU.
10. Verify CPU and GPU results and compare execution times.

## 7. Parameters

| Parameter | Value |
|---|---|
| Damping Factor | 0.85 |
| Convergence Tolerance | 1e-6 |
| Maximum Iterations | 100 |
| Verification Tolerance | 1e-6 |
| Threads per Block | 64, 128, 256, 512, 1024 |

## 8. Requirements

- NVIDIA GPU supporting CUDA.
- NVIDIA CUDA Toolkit.
- CUDA-compatible GPU drivers.
- C++ compiler.

## 9. Compilation

Navigate to the PageRank directory:

```bash
cd PageRank
```

Compile the CUDA program:

```bash
nvcc -O3 -arch=sm_70 pagerank.cu -o pagerank
```

Replace `sm_70` with the appropriate architecture for your GPU if required.

## 10. Execution

### Run using an input graph:

```bash
./pagerank graph.txt
```

### Run using a randomly generated graph:

```bash
./pagerank --random 1000000 8
```

This generates a random graph containing 1,000,000 vertices with an average outgoing degree of 8.

### Input Format

The input graph should contain directed edges, with one edge per line:

```text
0 1
0 2
1 3
2 3
3 4
```

Each line represents an edge from the source vertex to the destination vertex.

Lines beginning with `#` or `%` are treated as comments.

## 11. Performance Analysis

The implementation compares CPU and GPU execution performance using:

- CPU execution time.
- GPU kernel execution time.
- GPU end-to-end execution time.
- GPU speedup.
- Maximum CPU/GPU result difference.

The implementation tests different threads-per-block configurations: 64, 128, 256, 512, and 1024.

## 12. Correctness Verification

The GPU PageRank results are compared against the sequential CPU reference implementation.

The maximum absolute difference between CPU and GPU results is calculated to verify correctness.

The sum of all PageRank values is also checked to ensure it is approximately equal to 1.

## 13. Output

The program displays:

- Number of vertices and edges.
- Number of dangling vertices.
- GPU device information.
- CPU execution time.
- GPU execution time.
- Speedup for different thread configurations.
- CPU/GPU verification status.
- Top 5 vertices based on PageRank values.

## 14. Author

Group 14 – GPU Programming Project