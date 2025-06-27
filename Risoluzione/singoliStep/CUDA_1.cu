// maze_bfs_cuda.cu (v1.2)
// -----------------------------------------------------------------------------
// Parallellizzazione CUDA della versione sequenziale BFS.
// * Il nome del file dei labirinti è hard‑coded a "mazes.txt" (come nel codice
//   sequenziale originale). Nessun argomento da riga‑comando.
// * Si caricano *tutti* i labirinti definiti nel file (separati da righe vuote),
//   si stampa il labirinto, la sua versione numerata e il percorso più breve
//   (stessa semantica di output della versione CPU), ma sfruttando la GPU per
//   la BFS.
// * Un thread per nodo. Nessuna ottimizzazione: global memory, host‑side scan
//   per controllare la frontiera vuota, atomicCAS su visited e parent.
//
// Compilazione (Windows/MSVC + NVCC):
//     nvcc -std=c++17 -O2 -o maze_bfs_cuda.exe maze_bfs_cuda.cu
// -----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <chrono>   

// ----- GPU BFS kernel ---------------------------------------------------------
__global__ void bfsKernel(const char* maze,
                          int width,
                          int height,
                          const int* frontierFlag,
                          int* nextFrontierFlag,
                          int* visited,
                          int* parent,
                          bool* foundGoal,
                          int goalIdx)
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
        const int dr[4] = { -1,  1,  0,  0 };
        const int dc[4] = {  0,  0, -1,  1 };

        for (int k = 0; k < 4; ++k) {
            int nr = r + dr[k];
            int nc = c + dc[k];
            if (nr < 0 || nr >= height || nc < 0 || nc >= width) continue;
            int nidx = nr * width + nc;

            // muro = '#'. Qualsiasi altro carattere è percorribile.
            if (maze[nidx] != '#') {
                if (atomicCAS(&visited[nidx], 0, 1) == 0) {
                    parent[nidx] = idx;           // imposta il padre
                    nextFrontierFlag[nidx] = 1;   // push nella prossima frontiera
                }
            }
        }
    }
}

// ----- Driver GPU ‑ restituisce true se trovato e riporta il parent array -----
bool solveMazeCUDA(const char* h_maze,
                   int width,
                   int height,
                   int startIdx,
                   int goalIdx,
                   std::vector<int>& h_parent)
{
    const int numNodes   = width * height;
    const size_t bytesCh = numNodes * sizeof(char);
    const size_t bytesI  = numNodes * sizeof(int);

    // Device buffers
    char* d_maze          = nullptr;
    int*  d_frontier      = nullptr;
    int*  d_nextFrontier  = nullptr;
    int*  d_visited       = nullptr;
    int*  d_parent        = nullptr;
    bool* d_foundGoal     = nullptr;

    cudaMalloc(&d_maze,         bytesCh);
    cudaMalloc(&d_frontier,     bytesI);
    cudaMalloc(&d_nextFrontier, bytesI);
    cudaMalloc(&d_visited,      bytesI);
    cudaMalloc(&d_parent,       bytesI);
    cudaMalloc(&d_foundGoal,    sizeof(bool));

    cudaMemcpy(d_maze, h_maze, bytesCh, cudaMemcpyHostToDevice);
    cudaMemset(d_frontier,     0, bytesI);
    cudaMemset(d_nextFrontier, 0, bytesI);
    cudaMemset(d_visited,      0, bytesI);
    cudaMemset(d_parent,      -1, bytesI);          // tutti i parent = -1

    // Inizializza nodo di partenza
    cudaMemset(d_frontier  + startIdx, 1, sizeof(int));
    cudaMemset(d_visited   + startIdx, 1, sizeof(int));
    cudaMemcpy(d_parent + startIdx, &startIdx, sizeof(int), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocks          = (numNodes + threadsPerBlock - 1) / threadsPerBlock;

    bool h_foundGoal = false;
    bool continueBFS = true;

    std::vector<int> h_next(numNodes);
    h_parent.resize(numNodes, -1);

    while (continueBFS && !h_foundGoal) {
        cudaMemset(d_foundGoal, 0, sizeof(bool));

        bfsKernel<<<blocks, threadsPerBlock>>>(d_maze, width, height,
                                               d_frontier, d_nextFrontier,
                                               d_visited, d_parent, d_foundGoal, goalIdx);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_foundGoal, d_foundGoal, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_next.data(), d_nextFrontier, bytesI, cudaMemcpyDeviceToHost);

        // controlla se esiste almeno un nodo nella prossima frontiera
        continueBFS = std::any_of(h_next.begin(), h_next.end(), [](int v){ return v != 0; });

        if (continueBFS) {
            std::swap(d_frontier, d_nextFrontier);
            cudaMemset(d_nextFrontier, 0, bytesI);
        }
    }

    if (h_foundGoal) {
        cudaMemcpy(h_parent.data(), d_parent, bytesI, cudaMemcpyDeviceToHost);
    }

    cudaFree(d_maze);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_visited);
    cudaFree(d_parent);
    cudaFree(d_foundGoal);

    return h_foundGoal;
}

static int computePathLength(const std::vector<int>& parent,int startIdx,int goalIdx){
    if(parent.empty()||parent[goalIdx]==-1) return -1;
    int len=0,cur=goalIdx; while(cur!=startIdx){ cur=parent[cur]; if(cur==-1) return -1; ++len; }
    return len; 
}

// ----- Struttura dati labirinto (host) ----------------------------------------
struct MazeGPU {
    int rows = 0;
    int cols = 0;
    std::vector<std::string> lines;  // righe originali per stampa
    std::vector<char> cells;         // griglia lineare per la GPU
    int startIdx = -1;
    int goalIdx  = -1;
};

// ----- Loader mazes.txt (più labirinti separati da riga vuota) ----------------
bool loadMazesFromFile(const std::string& filename, std::vector<MazeGPU>& mazes)
{
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Errore apertura file: " << filename << "\n";
        return false;
    }

    std::cout << "Starting to read mazes from file." << std::endl;

    std::string line;
    std::vector<std::string> currentLines;

    auto finishCurrent = [&]() {
        if (currentLines.empty()) return;

        MazeGPU m;
        m.rows = static_cast<int>(currentLines.size());
        m.cols = static_cast<int>(currentLines.front().size());

        // verifica rettangolarità
        for (const auto& l : currentLines) {
            if (static_cast<int>(l.size()) != m.cols) {
                std::cerr << "Maze non rettangolare, skip." << std::endl;
                currentLines.clear();
                return;
            }
        }

        m.lines = currentLines;
        m.cells.resize(m.rows * m.cols);

        for (int r = 0; r < m.rows; ++r) {
            for (int c = 0; c < m.cols; ++c) {
                char ch = currentLines[r][c];
                m.cells[r * m.cols + c] = ch;
                if (ch == '2') m.startIdx = r * m.cols + c;
                if (ch == '3') m.goalIdx  = r * m.cols + c;
            }
        }

        if (m.startIdx == -1 || m.goalIdx == -1) {
            std::cerr << "Maze senza start/goal, skip." << std::endl;
        } else {
            mazes.emplace_back(std::move(m));
        }
        currentLines.clear();
    };

    int lineNum = 0;
    while (std::getline(in, line)) {
        // strip CR (Windows)
        if (!line.empty() && line.back() == '\r') line.pop_back();

        if (line.empty()) {
            finishCurrent();
        } else {
            std::cout << "Read line " << lineNum++ << ": '" << line << "'" << std::endl;
            currentLines.emplace_back(line);
        }
    }
    finishCurrent();

    std::cout << "Total mazes loaded: " << mazes.size() << std::endl;
    return !mazes.empty();
}

// ----- Funzioni di stampa (replica versioni C) --------------------------------
void printMaze(const MazeGPU& m)
{
    std::puts("\nLabirinto iniziale:");
    for (const auto& row : m.lines) {
        for (char ch : row) {
            std::putchar(ch);
            std::putchar(' ');
        }
        std::putchar('\n');
    }
}

void printMazeNumbered(const MazeGPU& m)
{
    std::puts("\nLabirinto numerato:");
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            int nodeId = r * m.cols + c;
            if (m.cells[nodeId] == '#') {
                std::printf("### ");
            } else {
                std::printf("%3d ", nodeId);
            }
        }
        std::putchar('\n');
    }
}

// ----- Ricostruisci e stampa percorso -----------------------------------------
void printPath(const std::vector<int>& parent, int startIdx, int goalIdx)
{
    std::vector<int> path;
    int current = goalIdx;
    while (current != -1 && current != startIdx) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(startIdx);
    std::reverse(path.begin(), path.end());

    std::cout << "\nPercorso più breve (sequenza di nodi): ";
    for (size_t i = 0; i < path.size(); ++i) {
        std::cout << path[i];
        if (i + 1 < path.size()) std::cout << ", ";
    }
    std::cout << "\n";
}

// ----- main -------------------------------------------------------------------
int main(){
    const char* filename="mazes100.txt";  // nome hard‑coded come da requisito

    // ------------------- CARICAMENTO LABIRINTI (codice originale) ------------
    std::vector<MazeGPU> mazes;
    if(!loadMazesFromFile(filename,mazes)){
        std::cerr<<"No mazes loaded from file. Exiting."<<std::endl; return 1; }

    // -------- NUOVE VARIABILI RIEPILOGO --------------------------------------
    std::vector<int> pathLengths(mazes.size(),-1);   // NEW
    int solvedCnt=0;                                 // NEW
    auto t0=std::chrono::high_resolution_clock::now(); // NEW: start timer
    // -------------------------------------------------------------------------

    for(size_t i=0;i<mazes.size();++i){
        const auto& m=mazes[i];
        std::vector<int> parent;
        bool reached=solveMazeCUDA(m.cells.data(),m.cols,m.rows,
                                   m.startIdx,m.goalIdx,parent);
        if(reached){ ++solvedCnt;                      // NEW
            pathLengths[i]=computePathLength(parent,m.startIdx,m.goalIdx); }
    }

    auto t1=std::chrono::high_resolution_clock::now();  // NEW
    double elapsedMs=std::chrono::duration<double,std::milli>(t1-t0).count(); // NEW

    // ----------------- OUTPUT RIEPILOGO --------------------------------------
    std::cout<<"\n--- RIEPILOGO ---\n";                      // NEW
    std::cout<<"Tempo totale: "<<elapsedMs<<" ms\n";       // NEW
    std::cout<<"Labirinti risolti: "<<solvedCnt<<"/"        // NEW
             <<mazes.size()<<"\n";                           // NEW
    std::cout<<"Lunghezze percorsi -> ";                     // NEW
    for(size_t i=0;i<pathLengths.size();++i){                // NEW
        std::cout<<"L"<<i+1<<":"<<pathLengths[i];
        if(i+1<pathLengths.size()) std::cout<<", "; }
    std::cout<<"\n";                                         // NEW

    return 0; 
}
