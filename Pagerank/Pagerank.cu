// ============================================================================
//  GPU Programming Project - Group 14
//  Parallel Implementation of PageRank on GPU (CUDA)
//
//  Workflow (as in the project abstract):
//    Graph Input -> CSR Construction -> CPU Reference Implementation
//    -> GPU Memory Transfer -> PageRank Initialization -> CUDA Parallelization
//    -> Parallel Rank Computation -> Convergence Check -> GPU Result
//    -> CPU/GPU Verification -> Performance Analysis
//
//  Formula (damping d = 0.85, N vertices):
//    PR_new(v) = (1-d)/N + d * ( danglingSum/N + sum_{u -> v} PR_old(u)/outdeg(u) )
//  danglingSum = total rank of vertices with no outgoing edges, spread evenly
//  so that the ranks always sum to 1.
//
//  Graph storage: CSR over INCOMING edges
//    in_row_ptr[v] .. in_row_ptr[v+1]-1  -> positions in in_col
//    in_col[e]                           -> source vertex u of edge u -> v
//  plus out_deg[u] for every vertex.
//
//  Build : nvcc -O3 -arch=sm_70 pagerank.cu -o pagerank
//          (needs sm_60 or newer for atomicAdd on double; use the arch of your GPU)
//  Run   : ./pagerank graph.txt            (edge list: "src dst" per line, '#' = comment)
//          ./pagerank --random 1000000 8   (random graph: N vertices, avg out-degree 8)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

const double DAMPING  = 0.85;
const double TOL      = 1e-6;   // convergence: L1 norm of (PR_new - PR_old)
const int    MAX_ITER = 100;
const double VERIFY_EPS = 1e-6; // max allowed |CPU - GPU| per vertex


// Graph in CSR (incoming edges)

struct Graph {
    int n = 0;
    int m = 0;
    std::vector<int> in_row_ptr;  // size n+1
    std::vector<int> in_col;      // size m (source vertices)
    std::vector<int> out_deg;     // size n
};


// Graph input

static void load_edge_list(const char* path, std::vector<std::pair<int,int>>& edges, int& n)
{
    std::ifstream in(path);
    if (!in) { fprintf(stderr, "Cannot open %s\n", path); exit(EXIT_FAILURE); }
    std::string line;
    int max_id = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream ss(line);
        long long u, v;
        if (!(ss >> u >> v)) continue;
        if (u < 0 || v < 0) continue;
        edges.push_back({(int)u, (int)v});
        max_id = std::max(max_id, (int)std::max(u, v));
    }
    n = max_id + 1;   // vertex ids 0..max_id (unused ids become dangling vertices)
}

static void random_graph(int n, int avg_deg, std::vector<std::pair<int,int>>& edges)
{
    std::mt19937 rng(42);   // fixed seed -> reproducible runs
    std::uniform_int_distribution<int> pick(0, n - 1);
    std::uniform_int_distribution<int> deg(0, 2 * avg_deg);   // mean = avg_deg, some 0 -> dangling
    edges.reserve((size_t)n * avg_deg);
    for (int u = 0; u < n; ++u) {
        int d = deg(rng);
        for (int k = 0; k < d; ++k) {
            int v = pick(rng);
            if (v != u) edges.push_back({u, v});
        }
    }
}


// CSR construction (incoming edges)

static Graph build_csr(int n, const std::vector<std::pair<int,int>>& edges)
{
    Graph g;
    g.n = n;
    g.m = (int)edges.size();
    g.in_row_ptr.assign(n + 1, 0);
    g.out_deg.assign(n, 0);
    g.in_col.resize(g.m);

    for (const auto& e : edges) {
        g.out_deg[e.first]++;
        g.in_row_ptr[e.second + 1]++;
    }
    for (int v = 0; v < n; ++v) g.in_row_ptr[v + 1] += g.in_row_ptr[v];

    std::vector<int> pos(g.in_row_ptr.begin(), g.in_row_ptr.end() - 1);
    for (const auto& e : edges) g.in_col[pos[e.second]++] = e.first;
    return g;
}


// CPU reference implementation (serial)

static int cpu_pagerank(const Graph& g, std::vector<double>& pr, double& final_err)
{
    const int n = g.n;
    pr.assign(n, 1.0 / n);
    std::vector<double> next(n), contrib(n);
    int iters = 0;
    final_err = 0.0;

    while (iters < MAX_ITER) {
        // contribution of every vertex + rank held by dangling vertices
        double dangling = 0.0;
        for (int u = 0; u < n; ++u) {
            if (g.out_deg[u] > 0) contrib[u] = pr[u] / g.out_deg[u];
            else { contrib[u] = 0.0; dangling += pr[u]; }
        }
        const double base = (1.0 - DAMPING) / n + DAMPING * dangling / n;

        // new rank from incoming neighbours
        double err = 0.0;
        for (int v = 0; v < n; ++v) {
            double sum = 0.0;
            for (int e = g.in_row_ptr[v]; e < g.in_row_ptr[v + 1]; ++e)
                sum += contrib[g.in_col[e]];
            next[v] = base + DAMPING * sum;
            err += fabs(next[v] - pr[v]);
        }
        pr.swap(next);
        ++iters;
        final_err = err;
        if (err < TOL) break;
    }
    return iters;
}


// CUDA kernels  (one thread = one vertex)


// PageRank initialization: PR(v) = 1/N
__global__ void init_kernel(double* pr, int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) pr[v] = 1.0 / n;
}

// contrib[u] = PR(u)/outdeg(u); dangling rank summed with a block reduction
__global__ void contrib_kernel(const double* __restrict__ pr,
                               const int*    __restrict__ out_deg,
                               double*       __restrict__ contrib,
                               double*       dangling_sum,
                               int n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int u   = blockIdx.x * blockDim.x + tid;

    double d = 0.0;
    if (u < n) {
        int deg = out_deg[u];
        if (deg > 0) contrib[u] = pr[u] / deg;
        else       { contrib[u] = 0.0; d = pr[u]; }
    }
    sdata[tid] = d;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {   // blockDim is a power of 2
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0 && sdata[0] != 0.0) atomicAdd(dangling_sum, sdata[0]);
}

// Parallel rank computation: each thread walks the incoming CSR edges of its
// vertex, reads only the previous iteration's values (pr_old / contrib) and
// writes into pr_new  -> two buffers keep the iterations synchronous.
// The L1 difference is reduced per block for the convergence check.
__global__ void update_kernel(const int*    __restrict__ in_row_ptr,
                              const int*    __restrict__ in_col,
                              const double* __restrict__ contrib,
                              const double* __restrict__ pr_old,
                              double*       __restrict__ pr_new,
                              const double* dangling_sum,
                              double*       l1_err,
                              int n, double damping)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int v   = blockIdx.x * blockDim.x + tid;

    double diff = 0.0;
    if (v < n) {
        double sum = 0.0;
        int start = in_row_ptr[v], end = in_row_ptr[v + 1];
        for (int e = start; e < end; ++e)
            sum += contrib[in_col[e]];

        double base = (1.0 - damping) / n + damping * (*dangling_sum) / n;
        double nv   = base + damping * sum;
        pr_new[v]   = nv;
        diff        = fabs(nv - pr_old[v]);
    }
    sdata[tid] = diff;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0 && sdata[0] != 0.0) atomicAdd(l1_err, sdata[0]);
}


// GPU driver

struct GpuResult {
    int    iters;
    double final_err;
    double kernel_ms;   // time spent inside the kernels only
    double e2e_ms;      // malloc + H2D + all iterations + D2H + free
};

static GpuResult gpu_pagerank(const Graph& g, int tpb, std::vector<double>& pr_out)
{
    GpuResult r{};
    const int n = g.n, m = g.m;
    const int blocks = (n + tpb - 1) / tpb;
    const size_t shmem = tpb * sizeof(double);

    cudaEvent_t k0, k1;
    CUDA_CHECK(cudaEventCreate(&k0));
    CUDA_CHECK(cudaEventCreate(&k1));

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- GPU memory allocation + transfer ----
    int *d_row_ptr, *d_col, *d_out_deg;
    double *d_pr_a, *d_pr_b, *d_contrib, *d_dangling, *d_err;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col,     std::max(m, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_deg, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pr_a,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pr_b,    n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_contrib, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dangling, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_err,      sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, g.in_row_ptr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (m > 0)
        CUDA_CHECK(cudaMemcpy(d_col, g.in_col.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_deg, g.out_deg.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // ---- PageRank initialization ----
    float ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(k0));
    init_kernel<<<blocks, tpb>>>(d_pr_a, n);
    CUDA_CHECK(cudaEventRecord(k1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(k1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, k0, k1));
    r.kernel_ms += ms;

    double* d_old = d_pr_a;
    double* d_new = d_pr_b;
    double h_err = 0.0;
    int iters = 0;

    // ---- iterate until convergence ----
    while (iters < MAX_ITER) {
        CUDA_CHECK(cudaMemset(d_dangling, 0, sizeof(double)));
        CUDA_CHECK(cudaMemset(d_err,      0, sizeof(double)));

        CUDA_CHECK(cudaEventRecord(k0));
        contrib_kernel<<<blocks, tpb, shmem>>>(d_old, d_out_deg, d_contrib, d_dangling, n);
        update_kernel <<<blocks, tpb, shmem>>>(d_row_ptr, d_col, d_contrib, d_old, d_new,
                                               d_dangling, d_err, n, DAMPING);
        CUDA_CHECK(cudaEventRecord(k1));
        CUDA_CHECK(cudaGetLastError());

        // convergence check: copy back the single L1 error value
        CUDA_CHECK(cudaMemcpy(&h_err, d_err, sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventElapsedTime(&ms, k0, k1));
        r.kernel_ms += ms;

        std::swap(d_old, d_new);   // swap the two PageRank buffers
        ++iters;
        if (h_err < TOL) break;
    }

    // ---- GPU result back to host ----
    pr_out.resize(n);
    CUDA_CHECK(cudaMemcpy(pr_out.data(), d_old, n * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_out_deg)); CUDA_CHECK(cudaFree(d_pr_a));
    CUDA_CHECK(cudaFree(d_pr_b));    CUDA_CHECK(cudaFree(d_contrib));
    CUDA_CHECK(cudaFree(d_dangling)); CUDA_CHECK(cudaFree(d_err));

    auto t_end = std::chrono::high_resolution_clock::now();
    r.e2e_ms    = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    r.iters     = iters;
    r.final_err = h_err;

    CUDA_CHECK(cudaEventDestroy(k0));
    CUDA_CHECK(cudaEventDestroy(k1));
    return r;
}


// CPU/GPU verification

static bool verify(const std::vector<double>& cpu, const std::vector<double>& gpu, double& max_diff)
{
    max_diff = 0.0;
    for (size_t i = 0; i < cpu.size(); ++i)
        max_diff = std::max(max_diff, fabs(cpu[i] - gpu[i]));
    return max_diff < VERIFY_EPS;
}

static void print_top(const std::vector<double>& pr, int k)
{
    std::vector<int> idx(pr.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    k = std::min<int>(k, (int)idx.size());
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return pr[a] > pr[b]; });
    printf("Top %d vertices by PageRank:\n", k);
    for (int i = 0; i < k; ++i) printf("  %2d. vertex %-10d  %.8f\n", i + 1, idx[i], pr[idx[i]]);
}


// main: input -> CSR -> CPU reference -> GPU (64..1024 threads) -> verify -> report

int main(int argc, char** argv)
{
    std::vector<std::pair<int,int>> edges;
    int n = 0;

    if (argc >= 2 && strcmp(argv[1], "--random") == 0) {
        n = (argc >= 3) ? atoi(argv[2]) : 1000000;
        int avg = (argc >= 4) ? atoi(argv[3]) : 8;
        printf("Generating random graph: N = %d, avg out-degree = %d\n", n, avg);
        random_graph(n, avg, edges);
    } else if (argc >= 2) {
        printf("Loading edge list: %s\n", argv[1]);
        load_edge_list(argv[1], edges, n);
    } else {
        printf("Usage: %s <edge_list.txt>  |  %s --random N AVG_DEG\n", argv[0], argv[0]);
        printf("No input given, using --random 1000000 8\n");
        n = 1000000;
        random_graph(n, 8, edges);
    }
    if (n <= 0) { fprintf(stderr, "Empty graph\n"); return EXIT_FAILURE; }

    // CSR construction
    Graph g = build_csr(n, edges);
    edges.clear(); edges.shrink_to_fit();
    int dangling = 0;
    for (int u = 0; u < n; ++u) if (g.out_deg[u] == 0) ++dangling;
    printf("Vertices: %d   Edges: %d   Dangling vertices: %d\n", g.n, g.m, dangling);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SMs: %d)\n", prop.name, prop.multiProcessorCount);
    CUDA_CHECK(cudaFree(0));   // create the CUDA context before timing

    // CPU reference
    std::vector<double> cpu_pr;
    double cpu_err;
    auto c0 = std::chrono::high_resolution_clock::now();
    int cpu_iters = cpu_pagerank(g, cpu_pr, cpu_err);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
    printf("\nCPU: %d iterations, final L1 error = %.3e, time = %.3f ms\n",
           cpu_iters, cpu_err, cpu_ms);

    // GPU runs for 64..1024 threads per block
    const int tpbs[] = {64, 128, 256, 512, 1024};
    std::vector<double> gpu_pr;
    bool all_ok = true;

    printf("\n%-8s %-6s %-14s %-14s %-12s %-12s %-12s %-8s\n",
           "Threads", "Iters", "Kernel(ms)", "E2E(ms)", "Speedup(K)", "Speedup(E2E)", "MaxDiff", "Check");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int tpb : tpbs) {
        if (tpb > prop.maxThreadsPerBlock) continue;
        GpuResult r = gpu_pagerank(g, tpb, gpu_pr);
        double max_diff;
        bool ok = verify(cpu_pr, gpu_pr, max_diff);
        all_ok = all_ok && ok;
        printf("%-8d %-6d %-14.3f %-14.3f %-12.2f %-12.2f %-12.2e %-8s\n",
               tpb, r.iters, r.kernel_ms, r.e2e_ms,
               cpu_ms / r.kernel_ms, cpu_ms / r.e2e_ms, max_diff, ok ? "PASS" : "FAIL");
    }

    printf("\nVerification: %s\n\n", all_ok ? "all GPU results match the CPU reference"
                                            : "MISMATCH between CPU and GPU");
    print_top(cpu_pr, 5);

    double sum = 0.0;
    for (double x : gpu_pr) sum += x;
    printf("\nSum of GPU ranks = %.8f (should be ~1.0)\n", sum);

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
