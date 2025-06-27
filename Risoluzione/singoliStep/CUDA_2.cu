// maze_bfs_cuda.cu (v1.3)
// -----------------------------------------------------------------------------
// Fix «punto 1» (bug dei cudaMemset a 1) e aggiunta statistiche finali.
//   • correzione inizializzazione start‑node (frontier & visited) con cudaMemcpy.
//   • macro CUDA_CHECK per error‑handling minimale.
//   • misurazione del tempo totale (std::chrono, in ms) *dopo* il parsing
//     e *prima* del solve dell’ultimo labirinto.
//   • conteggio labirinti risolti e lunghezza del percorso (‑1 se non risolto).
//   • stampa riepilogo: «Tempo totale: xx ms», «Risolti: n/m» e lista
//     «L1:18, L2:‑1, …».
// -----------------------------------------------------------------------------
// Compilazione:
//    nvcc -std=c++17 -O2 -o maze_bfs_cuda.exe maze_bfs_cuda.cu
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

// -----------------------------------------------------------------------------
// Macro di controllo errori -----------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// -----------------------------------------------------------------------------
// GPU BFS kernel ---------------------------------------------------------------
__global__ void bfsKernel(const char* maze, int width, int height,
                          const int* frontierFlag, int* nextFrontierFlag,
                          int* visited, int* parent,
                          bool* foundGoal, int goalIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    if (frontierFlag[idx]) {
        if (idx == goalIdx) {
            *foundGoal = true;
        }

        int r = idx / width;
        int c = idx % width;
        const int dr[4] = { -1, 1, 0, 0 };
        const int dc[4] = { 0, 0, -1, 1 };

        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
            int nidx = nr * width + nc;

            if (maze[nidx] != '#') { // muro = '#'
                if (atomicCAS(&visited[nidx], 0, 1) == 0) {
                    parent[nidx] = idx;
                    nextFrontierFlag[nidx] = 1;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Driver GPU -------------------------------------------------------------------
bool solveMazeCUDA(const char* h_maze, int width, int height,
                   int startIdx, int goalIdx, std::vector<int>& h_parent)
{
    const int numNodes   = width * height;
    const size_t bytesCh = numNodes * sizeof(char);
    const size_t bytesI  = numNodes * sizeof(int);

    char* d_maze = nullptr;
    int *d_frontier = nullptr, *d_nextFrontier = nullptr, *d_visited = nullptr, *d_parent = nullptr;
    bool* d_foundGoal = nullptr;

    CUDA_CHECK(cudaMalloc(&d_maze, bytesCh));
    CUDA_CHECK(cudaMalloc(&d_frontier, bytesI));
    CUDA_CHECK(cudaMalloc(&d_nextFrontier, bytesI));
    CUDA_CHECK(cudaMalloc(&d_visited, bytesI));
    CUDA_CHECK(cudaMalloc(&d_parent, bytesI));
    CUDA_CHECK(cudaMalloc(&d_foundGoal, sizeof(bool)));

    CUDA_CHECK(cudaMemcpy(d_maze, h_maze, bytesCh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_frontier, 0, bytesI));
    CUDA_CHECK(cudaMemset(d_nextFrontier, 0, bytesI));
    CUDA_CHECK(cudaMemset(d_visited, 0, bytesI));
    CUDA_CHECK(cudaMemset(d_parent, 0xFF, bytesI)); // -1 in little‑endian

    // inizializza start node (fix memset bug)
    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_frontier + startIdx, &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_visited  + startIdx, &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent   + startIdx, &startIdx, sizeof(int), cudaMemcpyHostToDevice));

    const int TPB = 256;
    const int blocks = (numNodes + TPB - 1) / TPB;

    bool h_foundGoal = false;
    bool continueBFS = true;
    std::vector<int> h_next(numNodes);
    h_parent.assign(numNodes, -1);

    while (continueBFS && !h_foundGoal) {
        CUDA_CHECK(cudaMemset(d_foundGoal, 0, sizeof(bool)));

        bfsKernel<<<blocks, TPB>>>(d_maze, width, height,
                                   d_frontier, d_nextFrontier,
                                   d_visited, d_parent, d_foundGoal, goalIdx);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&h_foundGoal, d_foundGoal, sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_next.data(), d_nextFrontier, bytesI, cudaMemcpyDeviceToHost));

        continueBFS = std::any_of(h_next.begin(), h_next.end(), [](int v) { return v != 0; });

        if (continueBFS) {
            std::swap(d_frontier, d_nextFrontier);
            CUDA_CHECK(cudaMemset(d_nextFrontier, 0, bytesI));
        }
    }

    if (h_foundGoal) {
        CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, bytesI, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_maze));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_nextFrontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_foundGoal));

    return h_foundGoal;
}

// -----------------------------------------------------------------------------
// Strutture e loader mazes ------------------------------------------------------
struct MazeGPU {
    int rows = 0, cols = 0;
    std::vector<std::string> lines;
    std::vector<char> cells;
    int startIdx = -1, goalIdx = -1;
};

bool loadMazesFromFile(const std::string& filename, std::vector<MazeGPU>& mazes)
{
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Errore apertura file " << filename << "\n";
        return false;
    }

    std::string line;
    std::vector<std::string> current;
    auto finish = [&]() {
        if (current.empty()) return;
        MazeGPU m; m.rows = (int)current.size(); m.cols = (int)current.front().size();
        for (auto& l : current) if ((int)l.size() != m.cols) { current.clear(); return; }
        m.lines = current; m.cells.resize(m.rows * m.cols);
        for (int r = 0; r < m.rows; ++r)
            for (int c = 0; c < m.cols; ++c) {
                char ch = current[r][c];
                m.cells[r * m.cols + c] = ch;
                if (ch == '2') m.startIdx = r * m.cols + c;
                if (ch == '3') m.goalIdx  = r * m.cols + c;
            }
        if (m.startIdx != -1 && m.goalIdx != -1) mazes.push_back(std::move(m));
        current.clear();
    };
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) finish(); else current.push_back(line);
    }
    finish();
    return !mazes.empty();
}

// -----------------------------------------------------------------------------
// Path reconstruction length ----------------------------------------------------
int computePathLength(const std::vector<int>& parent, int startIdx, int goalIdx)
{
    if (parent.empty() || parent[goalIdx] == -1) return -1;
    int len = 0, cur = goalIdx;
    while (cur != startIdx) {
        cur = parent[cur];
        if (cur == -1) return -1; // safety break (disconnected)
        ++len;
    }
    return len; // edges count
}

// -----------------------------------------------------------------------------
// main -------------------------------------------------------------------------
int main()
{
    const char* filename = "mazes100.txt";
    std::vector<MazeGPU> mazes;
    if (!loadMazesFromFile(filename, mazes)) {
        std::cerr << "Nessun labirinto caricato." << std::endl;
        return 1;
    }

    std::vector<int> pathLengths(mazes.size(), -1);
    int solvedCnt = 0;

    auto t0 = std::chrono::high_resolution_clock::now(); // start timing

    for (size_t i = 0; i < mazes.size(); ++i) {
        const auto& m = mazes[i];
        std::vector<int> parent;
        bool reached = solveMazeCUDA(m.cells.data(), m.cols, m.rows,
                                     m.startIdx, m.goalIdx, parent);
        if (reached) {
            ++solvedCnt;
            pathLengths[i] = computePathLength(parent, m.startIdx, m.goalIdx);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- output riepilogo ----
    std::cout << "\n--- RIEPILOGO ---\n";
    std::cout << "Tempo totale: " << elapsedMs << " ms\n";
    std::cout << "Labirinti risolti: " << solvedCnt << "/" << mazes.size() << "\n";
    std::cout << "Lunghezze percorsi -> ";
    for (size_t i = 0; i < pathLengths.size(); ++i) {
        std::cout << "L" << i + 1 << ":" << pathLengths[i];
        if (i + 1 < pathLengths.size()) std::cout << ", ";
    }
    std::cout << "\n";

    return 0;
}
