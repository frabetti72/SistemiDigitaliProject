// CUDA_2_Optimized.cu (v2.0) - Performance optimized version
// -----------------------------------------------------------------------------
// Optimizations based on simple.cu approach:
//   • Queue-based BFS instead of frontier flags
//   • Reduced memory transfers (only queue sizes)
//   • Atomic operations for queue management
//   • Improved memory coalescing
//   • Single kernel launch per level
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
// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

#define THREADS_PER_BLOCK 256
#define NOT_VISITED -1

// -----------------------------------------------------------------------------
// Optimized BFS kernel using queue approach
__global__ void bfsKernelOptimized(const char* maze, int width, int height,
                                   int* distance, int* parent,
                                   int* currentQueue, int currentQueueSize,
                                   int* nextQueue, int* nextQueueSize,
                                   int level, int goalIdx, bool* foundGoal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < currentQueueSize) {
        int currentNode = currentQueue[tid];
        
        // Check if we found the goal
        if (currentNode == goalIdx) {
            *foundGoal = true;
        }
        
        int r = currentNode / width;
        int c = currentNode % width;
        
        // Explore 4 directions
        const int dr[4] = {-1, 1, 0, 0};
        const int dc[4] = {0, 0, -1, 1};
        
        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            
            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                int neighbor = nr * width + nc;
                
                // Check if not a wall and not visited
                if (maze[neighbor] != '#' && distance[neighbor] == NOT_VISITED) {
                    // Use atomic compare-and-swap to claim this node
                    if (atomicCAS(&distance[neighbor], NOT_VISITED, level + 1) == NOT_VISITED) {
                        parent[neighbor] = currentNode;
                        // Add to next queue using atomic increment
                        int pos = atomicAdd(nextQueueSize, 1);
                        nextQueue[pos] = neighbor;
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Initialize arrays on GPU
__global__ void initializeArrays(int* distance, int* parent, int numNodes, int startIdx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numNodes) {
        if (tid == startIdx) {
            distance[tid] = 0;
            parent[tid] = startIdx;
        } else {
            distance[tid] = NOT_VISITED;
            parent[tid] = NOT_VISITED;
        }
    }
}

// -----------------------------------------------------------------------------
// Optimized CUDA solver
bool solveMazeCUDAOptimized(const char* h_maze, int width, int height,
                           int startIdx, int goalIdx, std::vector<int>& h_parent)
{
    const int numNodes = width * height;
    const size_t bytesCh = numNodes * sizeof(char);
    const size_t bytesI = numNodes * sizeof(int);
    
    // Device memory allocation
    char* d_maze = nullptr;
    int *d_distance = nullptr, *d_parent = nullptr;
    int *d_queue1 = nullptr, *d_queue2 = nullptr;
    int *d_nextQueueSize = nullptr;
    bool* d_foundGoal = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_maze, bytesCh));
    CUDA_CHECK(cudaMalloc(&d_distance, bytesI));
    CUDA_CHECK(cudaMalloc(&d_parent, bytesI));
    CUDA_CHECK(cudaMalloc(&d_queue1, bytesI));
    CUDA_CHECK(cudaMalloc(&d_queue2, bytesI));
    CUDA_CHECK(cudaMalloc(&d_nextQueueSize, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_foundGoal, sizeof(bool)));
    
    // Copy maze to device
    CUDA_CHECK(cudaMemcpy(d_maze, h_maze, bytesCh, cudaMemcpyHostToDevice));
    
    // Initialize arrays
    const int initBlocks = (numNodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    initializeArrays<<<initBlocks, THREADS_PER_BLOCK>>>(d_distance, d_parent, numNodes, startIdx);
    CUDA_CHECK(cudaGetLastError());
    
    // Initialize first queue with start node
    CUDA_CHECK(cudaMemcpy(d_queue1, &startIdx, sizeof(int), cudaMemcpyHostToDevice));
    
    int currentQueueSize = 1;
    int level = 0;
    bool h_foundGoal = false;
    
    // BFS main loop
    while (currentQueueSize > 0 && !h_foundGoal) {
        // Determine current and next queue pointers
        int* d_currentQueue = (level % 2 == 0) ? d_queue1 : d_queue2;
        int* d_nextQueue = (level % 2 == 0) ? d_queue2 : d_queue1;
        
        // Reset next queue size and found goal flag
        CUDA_CHECK(cudaMemset(d_nextQueueSize, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_foundGoal, 0, sizeof(bool)));
        
        // Launch BFS kernel
        const int blocks = (currentQueueSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        bfsKernelOptimized<<<blocks, THREADS_PER_BLOCK>>>(
            d_maze, width, height, d_distance, d_parent,
            d_currentQueue, currentQueueSize, d_nextQueue, d_nextQueueSize,
            level, goalIdx, d_foundGoal);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check if goal was found
        CUDA_CHECK(cudaMemcpy(&h_foundGoal, d_foundGoal, sizeof(bool), cudaMemcpyDeviceToHost));
        
        if (!h_foundGoal) {
            // Get next queue size for next iteration
            CUDA_CHECK(cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost));
        }
        
        level++;
    }
    
    // Copy results back if goal was found
    if (h_foundGoal) {
        h_parent.resize(numNodes);
        CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, bytesI, cudaMemcpyDeviceToHost));
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_maze));
    CUDA_CHECK(cudaFree(d_distance));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_queue1));
    CUDA_CHECK(cudaFree(d_queue2));
    CUDA_CHECK(cudaFree(d_nextQueueSize));
    CUDA_CHECK(cudaFree(d_foundGoal));
    
    return h_foundGoal;
}

// -----------------------------------------------------------------------------
// Maze structure and file loading (unchanged)
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
// Path length computation
int computePathLength(const std::vector<int>& parent, int startIdx, int goalIdx)
{
    if (parent.empty() || parent[goalIdx] == -1) return -1;
    int len = 0, cur = goalIdx;
    while (cur != startIdx) {
        cur = parent[cur];
        if (cur == -1) return -1;
        ++len;
    }
    return len;
}

// -----------------------------------------------------------------------------
// Main function
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

    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < mazes.size(); ++i) {
        const auto& m = mazes[i];
        std::vector<int> parent;
        bool reached = solveMazeCUDAOptimized(m.cells.data(), m.cols, m.rows,
                                            m.startIdx, m.goalIdx, parent);
        if (reached) {
            ++solvedCnt;
            pathLengths[i] = computePathLength(parent, m.startIdx, m.goalIdx);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Output summary
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