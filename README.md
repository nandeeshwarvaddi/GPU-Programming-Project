# GPU Programming Project: Parallel BFS, SSSP and PageRank

## Overview
This project implements three graph algorithms using NVIDIA CUDA:
1. **Breadth-First Search (BFS):** Parallel level-by-level graph traversal.
2. **Single-Source Shortest Path (SSSP):** Bellman-Ford shortest paths, including negative edge weights.
3. **PageRank:** Iterative vertex-importance computation using a damping factor.

## Technologies and Environment
- CUDA C/C++, NVIDIA CUDA Toolkit (`nvcc`)
- Google Colab, NVIDIA Tesla T4 GPU
- Compute capability 7.5; compilation architecture `sm_75`

## Repository Structure
- `BFS/` - CUDA BFS implementations and test graphs.
- `Pagerank/` - CUDA PageRank implementation.
- `SSSP-BellmanFord/` - CUDA Bellman-Ford implementation and test graphs.

## 1. Breadth-First Search (BFS)

### Implementation
- `BFS.cu`: Original static-graph implementation using an adjacency matrix.
- `BFS_FileInput.cu`: File-input implementation for evaluating different graphs.
- Uses current/next frontiers, a visited array, `atomicCAS()`, `atomicAdd()`, and level-synchronous kernel execution.
- The file-input version compares GPU distances against a CPU BFS reference.

### Compile and run
```bash
nvcc -O3 -arch=sm_75 BFS/BFS.cu -o bfs
nvcc -O3 -arch=sm_75 BFS/BFS_FileInput.cu -o bfs_file
./bfs
./bfs_file BFS/Testfiles/graph_small.txt
```

### BFS Evaluation
| Graph | Vertices | Edges | Result |
|---|---:|---:|---|
| Small connected | 5 | 5 | PASS |
| Disconnected | 7 | 4 | PASS |
| Large chain | 1,000 | 999 | PASS |
| Random sparse | 10,000 | 40,000 | PASS |

All four file-input tests passed CPU/GPU distance comparison. The disconnected graph correctly identified unreachable vertices.

## 2. Single-Source Shortest Path (SSSP)

### Implementation
SSSP uses Bellman-Ford with CUDA parallel edge relaxation:
- CPU Bellman-Ford reference and CSR graph representation.
- CUDA relaxation kernel using `atomicMin()`.
- Double-buffered distance arrays and a changed flag for convergence.
- CPU/GPU distance comparison and CSR transfer verification.

### Compile and run
```bash
nvcc -O3 -arch=sm_75 SSSP-BellmanFord/sssp.cu -o sssp
./sssp SSSP-BellmanFord/Testfiles/Graph.txt
```

### SSSP Evaluation
Twelve graph files were tested, including small, large sparse, and negative-weight graphs. All passed CPU/GPU distance comparison and CSR transfer verification.

Selected Tesla T4 benchmark results:

| Graph | Vertices | Edges | CPU (ms) | GPU Kernel (ms) |
|---|---:|---:|---:|---:|
| Benchmark 1 | 5,000 | 29,999 | 0.254 | 2.764 |
| Benchmark 2 | 10,000 | 59,999 | 0.445 | 2.994 |
| Benchmark 3 | 25,000 | 149,999 | 1.202 | 4.130 |
| Benchmark 4 | 50,000 | 299,999 | 2.701 | 9.982 |

**Observation:** Results were correct, but the GPU implementation was slower than the CPU reference in these experiments. Iterative relaxation and GPU execution overhead affect performance.

**Limitation:** Explicit negative-cycle detection is not implemented.

## 3. PageRank

### Implementation
The CUDA PageRank implementation uses incoming-edge CSR representation:
- CPU sequential reference.
- CUDA initialization, contribution, and rank-update kernels.
- Shared-memory reduction for dangling-vertex rank.
- CPU/GPU numerical verification.

Parameters: damping factor 0.85, convergence tolerance `1e-6`, maximum 100 iterations. Block sizes tested: 64, 128, 256, 512, and 1024.

### Compile and run
```bash
nvcc -O3 -arch=sm_75 Pagerank/Pagerank.cu -o pagerank
./pagerank --random 1000 8
./pagerank --random 10000 8
./pagerank --random 100000 8
./pagerank --random 1000000 8
```
The arguments specify the number of vertices and average out-degree.

### PageRank Evaluation
All five tested block sizes passed CPU-reference verification.

| Vertices | Edges | CPU (ms) | GPU Kernel (ms) | GPU End-to-End (ms) |
|---:|---:|---:|---:|---:|
| 1,000 | 8,123 | 0.261 | 0.260 | 0.783 |
| 10,000 | 80,839 | 2.594 | 0.324 | 0.946 |
| 100,000 | 798,632 | 26.711 | 2.297 | 4.364 |
| 1,000,000 | 7,998,465 | 543.090 | 39.140 | 52.523 |

For one million vertices, the best measured configuration was 1,024 threads per block:
- Kernel speedup: approximately 13.88x.
- End-to-end speedup: approximately 10.34x.
- GPU rank sum: approximately 1.0.
- CPU/GPU verification: PASS.

## Overall Evaluation Summary
| Algorithm | Evaluation | Correctness |
|---|---|---|
| BFS | Four file-input graphs | All passed |
| SSSP | Twelve graph files | All passed |
| PageRank | Four graph sizes and five block sizes | All passed |

## Limitations and Future Improvements
- Implement CSR-based BFS for large sparse graphs.
- Improve GPU performance for SSSP.
- Add negative-cycle detection to SSSP.
- Reduce synchronization and memory-transfer overhead.
- Evaluate more graph structures and densities.
- Test additional CUDA architectures.
- Repeat timing experiments for more stable measurements.

## Contributors
This is a collaborative academic GPU Programming project. Contributions include algorithm implementation, CUDA development, testing, performance evaluation, and documentation. Refer to the repository commit history and pull requests for individual contributions.

## Conclusion
This project demonstrates GPU implementations of BFS, Bellman-Ford SSSP, and PageRank. The experiments show that GPU acceleration depends on graph size, graph structure, algorithm design, and execution overhead. PageRank achieved significant speedups on larger graphs, while the tested SSSP implementation remained slower than its CPU reference.
