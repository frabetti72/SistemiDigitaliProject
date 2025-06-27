//versione con doppia velocità rispetto a CUDA 2
//varie ottimizzazzioni sparse
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)

// Strutture e loader mazes
struct MazeGPU {
    int rows = 0, cols = 0;
    std::vector<std::string> lines;
    std::vector<char> cells;
    int startIdx = -1, goalIdx = -1;
};

bool loadMazesFromFile(const std::string& filename, std::vector<MazeGPU>& mazes) {
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

// Path reconstruction length
int computePathLength(const std::vector<int>& parent, int startIdx, int goalIdx) {
    if (parent.empty() || parent[goalIdx] == -1) return -1;
    int len = 0, cur = goalIdx;
    while (cur != startIdx) {
        cur = parent[cur];
        if (cur == -1) return -1;
        ++len;
    }
    return len;
}

// Kernel ottimizzato che processa la frontiera corrente
__global__ void bfsKernelWorkEfficient(
    const char* __restrict__ maze, 
    int width, int height,
    const int* __restrict__ currentFrontier,
    int* __restrict__ nextFrontier,
    int* __restrict__ visited,
    int* __restrict__ parent,
    int currentFrontierSize,
    int* __restrict__ nextFrontierSize,
    bool* __restrict__ foundGoal,
    int goalIdx) 
{
    // Usa grid-stride loop per processare tutti gli elementi
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Direzioni di movimento
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    
    // Grid-stride loop: ogni thread può processare più nodi
    for (int i = tid; i < currentFrontierSize; i += stride) {
        int idx = currentFrontier[i];
        
        if (idx == goalIdx) {
            *foundGoal = true;
        }
        
        int r = idx / width;
        int c = idx % width;
        
        // Esplora i vicini
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            
            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                int nidx = nr * width + nc;
                
                if (maze[nidx] != '#') {
                    // Usa atomicCAS per marcare come visitato
                    if (atomicCAS(&visited[nidx], 0, 1) == 0) {
                        parent[nidx] = idx;
                        // Aggiungi alla prossima frontiera
                        int pos = atomicAdd(nextFrontierSize, 1);
                        if (pos < width * height) {
                            nextFrontier[pos] = nidx;
                        }
                    }
                }
            }
        }
    }
}

// Driver GPU ottimizzato
bool solveMazeCUDA(const char* h_maze, int width, int height,
                   int startIdx, int goalIdx, std::vector<int>& h_parent) {
    const int numNodes = width * height;
    
    // Allocazioni GPU
    char* d_maze;
    int *d_currentFrontier, *d_nextFrontier;
    int *d_visited, *d_parent;
    int *d_nextFrontierSize;
    bool* d_foundGoal;
    
    CUDA_CHECK(cudaMalloc(&d_maze, numNodes * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_currentFrontier, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontier, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_parent, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontierSize, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_foundGoal, sizeof(bool)));
    
    // Inizializzazione
    CUDA_CHECK(cudaMemcpy(d_maze, h_maze, numNodes * sizeof(char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_visited, 0, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_parent, 0xFF, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_foundGoal, 0, sizeof(bool)));
    
    // Setup iniziale - solo il nodo di partenza nella frontiera
    CUDA_CHECK(cudaMemcpy(d_currentFrontier, &startIdx, sizeof(int), cudaMemcpyHostToDevice));
    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_visited + startIdx, &one, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent + startIdx, &startIdx, sizeof(int), cudaMemcpyHostToDevice));
    
    // Configurazione kernel ottimizzata per utilizzare più SM
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    const int THREADS_PER_BLOCK = 256;
    // Usa almeno tanti blocchi quanti sono gli SM disponibili
    const int MIN_BLOCKS = prop.multiProcessorCount;
    
    bool h_foundGoal = false;
    int currentFrontierSize = 1;
    int h_nextFrontierSize = 0;
    
    // Array per tenere traccia delle frontiere
    std::vector<int> h_currentFrontier(numNodes);
    h_currentFrontier[0] = startIdx;
    
    // BFS livello per livello
    while (currentFrontierSize > 0 && !h_foundGoal) {
        CUDA_CHECK(cudaMemset(d_nextFrontierSize, 0, sizeof(int)));
        
        // Calcola numero di blocchi basato sul lavoro disponibile
        // Ma usa sempre almeno MIN_BLOCKS per utilizzare tutti gli SM
        int requiredBlocks = (currentFrontierSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int blocks = std::max(MIN_BLOCKS, requiredBlocks);
        
        // Se abbiamo pochi nodi da processare, riduci i thread per blocco
        // per avere più blocchi attivi
        int threadsPerBlock = THREADS_PER_BLOCK;
        if (currentFrontierSize < MIN_BLOCKS * THREADS_PER_BLOCK / 4) {
            threadsPerBlock = std::max(32, currentFrontierSize / MIN_BLOCKS);
            threadsPerBlock = (threadsPerBlock + 31) / 32 * 32; // Arrotonda a multiplo di 32
            blocks = (currentFrontierSize + threadsPerBlock - 1) / threadsPerBlock;
            blocks = std::max(blocks, MIN_BLOCKS);
        }
        
        // Lancia kernel con configurazione ottimizzata
        bfsKernelWorkEfficient<<<blocks, threadsPerBlock>>>(
            d_maze, width, height,
            d_currentFrontier, d_nextFrontier,
            d_visited, d_parent,
            currentFrontierSize, d_nextFrontierSize,
            d_foundGoal, goalIdx
        );
        CUDA_CHECK(cudaGetLastError());
        
        // Copia solo le informazioni essenziali
        CUDA_CHECK(cudaMemcpy(&h_foundGoal, d_foundGoal, sizeof(bool), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_nextFrontierSize, d_nextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_nextFrontierSize > 0 && !h_foundGoal) {
            // Swap delle frontiere
            std::swap(d_currentFrontier, d_nextFrontier);
            currentFrontierSize = h_nextFrontierSize;
        } else {
            currentFrontierSize = 0;
        }
    }
    
    // Copia risultati
    if (h_foundGoal) {
        h_parent.resize(numNodes);
        CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, numNodes * sizeof(int), cudaMemcpyDeviceToHost));
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_maze));
    CUDA_CHECK(cudaFree(d_currentFrontier));
    CUDA_CHECK(cudaFree(d_nextFrontier));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_nextFrontierSize));
    CUDA_CHECK(cudaFree(d_foundGoal));
    
    return h_foundGoal;
}

int main() {
    const char* filename = "mazes100.txt";
    std::vector<MazeGPU> mazes;
    if (!loadMazesFromFile(filename, mazes)) {
        std::cerr << "Nessun labirinto caricato." << std::endl;
        return 1;
    }

    std::vector<int> pathLengths(mazes.size(), -1);
    int solvedCnt = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

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

    // Output riepilogo
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