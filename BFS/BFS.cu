#include <iostream>
#include <vector>
#include <cuda_runtime.h>
using namespace std;

// CUDA KERNEL: PARALLEL BFS
__global__ void bfsKernel(
    int *adj,
    int *visited,
    int *distance,
    int *frontier,
    int *nextFrontier,
    int frontierSize,
    int *nextFrontierSize,
    int V)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontierSize){
        int u = frontier[tid];

        // Explore neighbors of vertex u
        for (int v = 0; v < V; v++){
            if (adj[u * V + v] == 1){
                // Atomically mark vertex as visited
                if (atomicCAS(&visited[v], 0, 1) == 0){
                    distance[v] = distance[u] + 1;
                    int pos = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[pos] = v;
                }
            }
        }
    }
}


// MAIN FUNCTION


int main(){
    
    // GRAPH INPUTS - CHANGE HERE
    const int V = 5;          // Number of vertices
    const int E = 5;          // Number of edges
    const int source = 0;     // Starting vertex

    // Edge list (undirected graph)
    int edges[E][2] = {
        {0, 1},
        {0, 2},
        {1, 3},
        {1, 4},
        {2, 4}
    };

    // CREATE ADJACENCY MATRIX
    vector<int> adj(V * V, 0);
    for (int i = 0; i < E; i++){
        int u = edges[i][0];
        int v = edges[i][1];

        adj[u * V + v] = 1;
        adj[v * V + u] = 1;
    }  

    // INITIALIZE BFS ARRAYS
    vector<int> h_visited(V, 0);
    vector<int> h_distance(V, -1);
    vector<int> h_frontier(V, -1);

    h_visited[source] = 1;
    h_distance[source] = 0;
    h_frontier[0] = source;

    int frontierSize = 1;
    int nextFrontierSize = 0;

    // GPU MEMORY ALLOCATION
    int *d_adj, *d_visited, *d_distance;
    int *d_frontier, *d_nextFrontier;
    int *d_nextFrontierSize;

    cudaMalloc((void**)&d_adj, V * V * sizeof(int));
    cudaMalloc((void**)&d_visited, V * sizeof(int));
    cudaMalloc((void**)&d_distance, V * sizeof(int));

    cudaMalloc((void**)&d_frontier, V * sizeof(int));
    cudaMalloc((void**)&d_nextFrontier, V * sizeof(int));
    cudaMalloc((void**)&d_nextFrontierSize, sizeof(int));
 
    // COPY DATA FROM CPU TO GPU
    cudaMemcpy(d_adj, adj.data(),V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, h_distance.data(), V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_frontier, h_frontier.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    // PARALLEL BFS EXECUTION
    int threads = 256;

    while (frontierSize > 0){
        nextFrontierSize = 0;

        // Reset next frontier counter
        cudaMemcpy(d_nextFrontierSize,&nextFrontierSize,sizeof(int),cudaMemcpyHostToDevice);

        int blocks = (frontierSize + threads - 1) / threads;

        // Launch CUDA kernel
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

        cudaDeviceSynchronize();

        // Get next level size
        cudaMemcpy(&nextFrontierSize, d_nextFrontierSize,sizeof(int), cudaMemcpyDeviceToHost);
        // Swap current and next frontiers
        swap(d_frontier, d_nextFrontier);
        frontierSize = nextFrontierSize;
    }

    // COPY RESULTS FROM GPU TO CPU

    cudaMemcpy(h_distance.data(), d_distance,V * sizeof(int), cudaMemcpyDeviceToHost);

    // DISPLAY RESULTS
    cout << "\nParallel BFS using CUDA\n";
    cout << "Number of vertices: " << V << endl;
    cout << "Number of edges: " << E << endl;
    cout << "Source vertex: " << source << endl;
    cout << "\nBFS Distances:\n";

    for (int i = 0; i < V; i++){
        cout << "Vertex " << i
             << " : " << h_distance[i] << endl;
    }

    // FREE GPU MEMORY

    cudaFree(d_adj);
    cudaFree(d_visited);
    cudaFree(d_distance);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextFrontierSize);

    return 0;
}
