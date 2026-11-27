#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <algorithm>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) \
             << " at line " << __LINE__ << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void bfsKernel(
    const int* rowPtr, const int* col,
    int* visited, int* distance,
    const int* frontier, int frontierSize,
    int* nextFrontier, int* nextSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < frontierSize) {
        int u = frontier[tid];

        for (int e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
            int v = col[e];

            if (atomicCAS(&visited[v], 0, 1) == 0) {
                distance[v] = distance[u] + 1;
                int pos = atomicAdd(nextSize, 1);
                nextFrontier[pos] = v;
            }
        }
    }
}

bool readGraph(const string& filename, int& V, int& E,
               vector<int>& rowPtr, vector<int>& col)
{
    ifstream in(filename);
    if (!in) {
        cerr << "Cannot open input file: " << filename << endl;
        return false;
    }

    if (!(in >> V >> E) || V <= 0 || E < 0) {
        cerr << "Invalid graph header. Expected: V E" << endl;
        return false;
    }

    vector<pair<int,int>> edges;
    vector<int> degree(V, 0);

    for (int i = 0; i < E; i++) {
        int u, v;
        if (!(in >> u >> v) || u < 0 || u >= V || v < 0 || v >= V) {
            cerr << "Invalid or missing edge at line " << i + 2 << endl;
            return false;
        }

        edges.push_back({u, v});
        degree[u]++;
        if (u != v) degree[v]++;
    }

    rowPtr.assign(V + 1, 0);
    for (int i = 0; i < V; i++)
        rowPtr[i + 1] = rowPtr[i] + degree[i];

    col.resize(rowPtr[V]);
    vector<int> cursor = rowPtr;

    for (auto [u, v] : edges) {
        col[cursor[u]++] = v;
        if (u != v) col[cursor[v]++] = u;
    }

    return true;
}

vector<int> cpuBFS(int V, const vector<int>& rowPtr,
                   const vector<int>& col, int source)
{
    vector<int> dist(V, -1);
    queue<int> q;

    dist[source] = 0;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int e = rowPtr[u]; e < rowPtr[u + 1]; e++) {
            int v = col[e];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) {
        cout << "Usage: " << argv[0] << " <graph_file> [source]\n";
        return 1;
    }

    string filename = argv[1];
    int V, E;
    vector<int> rowPtr, col;

    if (!readGraph(filename, V, E, rowPtr, col))
        return 1;

    int source = (argc == 3) ? stoi(argv[2]) : 0;
    if (source < 0 || source >= V) {
        cerr << "Invalid source vertex\n";
        return 1;
    }

    vector<int> cpuDist = cpuBFS(V, rowPtr, col, source);

    int *d_rowPtr, *d_col, *d_visited, *d_dist;
    int *d_frontier, *d_next, *d_nextSize;

    CUDA_CHECK(cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, max<size_t>(1, col.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dist, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_frontier, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextSize, sizeof(int)));

    vector<int> initialVisited(V, 0), initialDist(V, -1);
    initialVisited[source] = 1;
    initialDist[source] = 0;

    CUDA_CHECK(cudaMemcpy(d_rowPtr, rowPtr.data(),
                          (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!col.empty())
        CUDA_CHECK(cudaMemcpy(d_col, col.data(),
                              col.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited, initialVisited.data(),
                          V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dist, initialDist.data(),
                          V * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier, &source, sizeof(int),
                          cudaMemcpyHostToDevice));

    int frontierSize = 1, nextSize = 0;
    const int threads = 256;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    while (frontierSize > 0) {
        CUDA_CHECK(cudaMemset(d_nextSize, 0, sizeof(int)));

        int blocks = (frontierSize + threads - 1) / threads;
        bfsKernel<<<blocks, threads>>>(
            d_rowPtr, d_col, d_visited, d_dist,
            d_frontier, frontierSize, d_next, d_nextSize);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&nextSize, d_nextSize, sizeof(int),
                              cudaMemcpyDeviceToHost));

        swap(d_frontier, d_next);
        frontierSize = nextSize;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));

    vector<int> gpuDist(V);
    CUDA_CHECK(cudaMemcpy(gpuDist.data(), d_dist,
                          V * sizeof(int), cudaMemcpyDeviceToHost));

    bool pass = (gpuDist == cpuDist);
    int reachable = count_if(gpuDist.begin(), gpuDist.end(),
                             [](int d) { return d != -1; });

    cout << "\nCUDA BFS File-Input Evaluation\n";
    cout << "Input file: " << filename << "\n";
    cout << "Vertices: " << V << " | Input edges: " << E << "\n";
    cout << "Source: " << source << "\n";
    cout << "Reachable vertices: " << reachable << "/" << V << "\n";
    cout << "GPU execution time: " << gpuMs << " ms\n";
    cout << "CPU/GPU distance comparison: "
         << (pass ? "PASS" : "FAIL") << "\n";

    if (!pass) {
        cout << "\nVertex | CPU distance | GPU distance\n";
        for (int i = 0; i < V; i++)
            if (cpuDist[i] != gpuDist[i])
                cout << i << " | " << cpuDist[i]
                     << " | " << gpuDist[i] << "\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaFree(d_nextSize));

    return pass ? 0 : 2;
}
