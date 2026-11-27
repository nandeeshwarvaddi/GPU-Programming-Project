#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

// Check CUDA errors
#define CUDA_CHECK(call) \{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
}
// 1. SEQUENTIAL BFS - CPU IMPLEMENTATION
void cpuBFS(const vector<int>& adj,vector<int>& distance,int V,int source){
    vector<int> visited(V, 0);
    queue<int> q;

    visited[source] = 1;
    distance[source] = 0;

    q.push(source);

    while (!q.empty()){
        int u = q.front();
        q.pop();

        // Explore neighbors sequentially
        for (int v = 0; v < V; v++){
            if (adj[u * V + v] == 1 && !visited[v]){
                visited[v] = 1;
                distance[v] = distance[u] + 1;
                q.push(v);
            }
        }
    }
}

// 2. PARALLEL BFS KERNEL - GPU IMPLEMENTATION
__global__ void bfsKernel(
    int *adj,
    int *visited,
    int *distance,
    int *frontier,
    int *nextFrontier,
    int frontierSize,
    int *nextFrontierSize,
    int V){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontierSize){
        int u = frontier[tid];

        // Explore neighbors of vertex u
        for (int v = 0; v < V; v++){
            if (adj[u * V + v] == 1){
                // Avoid duplicate visits
                if (atomicCAS(&visited[v], 0, 1) == 0){
                    distance[v] = distance[u] + 1;
                    // Insert vertex into next frontier
                    int pos = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[pos] = v;
                }
            }
        }
    }
}

// 3. PARALLEL BFS - GPU HOST FUNCTION
void gpuBFS(const vector<int>& adj,vector<int>& distance,int V,int source){
    vector<int> h_visited(V, 0);
    vector<int> h_frontier(V, -1);

    h_visited[source] = 1;
    distance[source] = 0;
    h_frontier[0] = source;

    int frontierSize = 1;
    int nextFrontierSize = 0;

    // GPU memory pointers
    int *d_adj, *d_visited, *d_distance;
    int *d_frontier, *d_nextFrontier;
    int *d_nextFrontierSize;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_adj, V * V * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_distance, V * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&d_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_nextFrontier, V * sizeof(int)));

    CUDA_CHECK(cudaMalloc((void**)&d_nextFrontierSize, sizeof(int)));

    // Copy initial data from CPU to GPU
    CUDA_CHECK(cudaMemcpy(d_adj, adj.data(), V * V * sizeof(int),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_visited, h_visited.data(), V * sizeof(int),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_distance, distance.data(), V * sizeof(int),cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_frontier, h_frontier.data(), V * sizeof(int),cudaMemcpyHostToDevice));

    int threads = 256;

    // BFS level-by-level execution
    while (frontierSize > 0){
        nextFrontierSize = 0;

        // Reset next frontier counter
        CUDA_CHECK(cudaMemcpy(d_nextFrontierSize,&nextFrontierSize,sizeof(int),cudaMemcpyHostToDevice));

        int blocks = (frontierSize + threads - 1) / threads;

        // Launch GPU kernel
        bfsKernel<<<blocks, threads>>>(
            d_adj,
            d_visited,
            d_distance,
            d_frontier,
            d_nextFrontier,
            frontierSize,
            d_nextFrontierSize,
            V
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Get next frontier size
        CUDA_CHECK(cudaMemcpy(&nextFrontierSize,d_nextFrontierSize,sizeof(int),cudaMemcpyDeviceToHost));

        // Swap frontiers
        swap(d_frontier, d_nextFrontier);
        frontierSize = nextFrontierSize;
    }

    // Copy final distances from GPU to CPU
    CUDA_CHECK(cudaMemcpy(
        distance.data(),
        d_distance,
        V * sizeof(int),
        cudaMemcpyDeviceToHost));

    // Free GPU memory
    cudaFree(d_adj);
    cudaFree(d_visited);
    cudaFree(d_distance);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);
}

// 4. MAIN FUNCTION
int main(){
    // Graph inputs
    const int V = 5;
    const int E = 5;
    const int source = 0;

    // Static undirected graph
    int edges[E][2] = {
        {0, 1},
        {0, 2},
        {1, 3},
        {1, 4},
        {2, 4}
    };

    // Create adjacency matrix
    vector<int> adj(V * V, 0);

    for (int i = 0; i < E; i++){
        int u = edges[i][0];
        int v = edges[i][1];

        adj[u * V + v] = 1;
        adj[v * V + u] = 1;
    }

    // Initialize distance arrays
    vector<int> cpu_distance(V, -1);
    vector<int> gpu_distance(V, -1);

    // CPU EXECUTION TIME
    auto cpuStart = high_resolution_clock::now();
    cpuBFS(adj, cpu_distance, V, source);
    auto cpuEnd = high_resolution_clock::now();
    double cpuTime =duration<double, milli>(cpuEnd - cpuStart).count();

    // GPU EXECUTION TIME
    // Includes GPU memory allocation, data transfer,
    // kernel execution and result transfer.

    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpuStart = high_resolution_clock::now();

    gpuBFS(adj, gpu_distance, V, source);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpuEnd = high_resolution_clock::now();

    double gpuTime = duration<double, milli>(gpuEnd - gpuStart).count();


    // DISPLAY CPU RESULTS

    cout << "\n========== CPU BFS ==========\n";

    for (int i = 0; i < V; i++){
        cout << "Vertex " << i
             << " : " << cpu_distance[i] << endl;
    }

    cout << "\nCPU Execution Time: "
         << cpuTime << " ms" << endl;


    // DISPLAY GPU RESULTS

    cout << "\n========== GPU BFS (CUDA) ==========\n";

    for (int i = 0; i < V; i++){
        cout << "Vertex " << i
             << " : " << gpu_distance[i] << endl;
    }

    cout << "\nGPU Execution Time: "
         << gpuTime << " ms" << endl;


    // COMPARE BOTH IMPLEMENTATIONS

    cout << "\n========== PERFORMANCE COMPARISON ==========\n";

    cout << "Number of vertices: " << V << endl;
    cout << "Number of edges: " << E << endl;
    cout << "Source vertex: " << source << endl;

    bool correct = (cpu_distance == gpu_distance);

    cout << "\nResult Verification: "
         << (correct ? "PASSED" : "FAILED") << endl;

    cout << "\nCPU Time: " << cpuTime << " ms";
    cout << "\nGPU Time: " << gpuTime << " ms" << endl;

    if (gpuTime > 0){
        double speedup = cpuTime / gpuTime;

        cout << "GPU Speedup: "
             << speedup << "x" << endl;
    }

    if (cpuTime < gpuTime)
        cout << "\nCPU is faster for this graph." << endl;
    else
        cout << "\nGPU is faster for this graph." << endl;

    return 0;
}
