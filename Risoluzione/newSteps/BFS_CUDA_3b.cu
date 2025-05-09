
#include <stdio.h>
#include <cuda_runtime.h>
#include <errno.h>

#define ROWS 50
#define COLS 50
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4
#define BLOCK_SIZE 256
#define MAX_MAZES 100 

struct MazeConfig {
    int rows;
    int cols;
    int maxNodes;
};

// Strutture dati per GPU
typedef struct {
    int row;
    int col;
} Point;

typedef struct {
    Point pos;
    bool visited;
    int nodeNum;
    int parentIndex;
    bool wall;
} Node;

// Costanti per le direzioni (da copiare in memoria costante GPU)
__constant__ Point directions[NUM_DIRECTIONS] = {
    {-1, 0},  // UP
    {1, 0},   // DOWN
    {0, -1},  // LEFT
    {0, 1}    // RIGHT
};

// Funzioni helper utilizzabili sia su host che device
__host__ __device__ bool isValid(int r, int c, const MazeConfig* config) {
    return r >= 0 && r < config->rows && c >= 0 && c < config->cols;
}

__host__ __device__ int coordToIndex(int row, int col, const MazeConfig* config) {
    return row * config->cols + col;
}

__host__ __device__ Point makePoint(int row, int col) {
    Point p;
    p.row = row;
    p.col = col;
    return p;
}
// Kernel per la ricostruzione parallela del percorso
__global__ void reconstructPathKernel(
    Node* nodes,
    int* path,
    int* pathLength,
    int startIdx,
    int endIdx,
    int maxLength
) {
    __shared__ int sharedPath[BLOCK_SIZE];
    int currentLength = 0;
    
    // Il thread 0 si occupa della ricostruzione iniziale
    if (threadIdx.x == 0) {
        int currentIdx = endIdx;
        while (currentIdx != startIdx && currentLength < maxLength) {
            sharedPath[currentLength++] = nodes[currentIdx].nodeNum;
            currentIdx = nodes[currentIdx].parentIndex;
        }
        if (currentLength < maxLength) {
            sharedPath[currentLength++] = nodes[startIdx].nodeNum;
        }
        *pathLength = currentLength;
    }
    
    // Sincronizza tutti i thread del blocco
    __syncthreads();
    
    // Parallelizza l'inversione del percorso
    int halfLength = *pathLength / 2;
    for (int i = threadIdx.x; i < halfLength; i += blockDim.x) {
        int temp = sharedPath[i];
        sharedPath[i] = sharedPath[*pathLength - 1 - i];
        sharedPath[*pathLength - 1 - i] = temp;
    }
    
    // Sincronizza prima della copia finale
    __syncthreads();
    
    // Copia il risultato in memoria globale
    for (int i = threadIdx.x; i < *pathLength; i += blockDim.x) {
        path[i] = sharedPath[i];
    }
}

// Kernel principale per l'esplorazione BFS
__global__ void exploreLevel(
    Node*           nodes,
    int*            frontier,
    int*            nextFrontier,
    int*            frontierSize,
    int*            nextFrontierSize,
    bool*           levelCompleted,
    int             endIdx,
    MazeConfig*     cfg
) {
    extern __shared__ int sharedFrontier[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    // Carica frontiera corrente in shared
    if (tid < *frontierSize) {
        sharedFrontier[localId] = frontier[tid];
    }
    __syncthreads();

    // Ogni thread espande il proprio nodo
    if (localId < *frontierSize) {
        int nodeIdx = sharedFrontier[localId];
        Point p = nodes[nodeIdx].pos;

        for (int d = 0; d < 4; ++d) {
            int nr = p.row + directions[d].row;
            int nc = p.col + directions[d].col;
            if (isValid(nr, nc, cfg)) {
                int newIdx = coordToIndex(nr, nc, cfg);
                if (!nodes[newIdx].visited && !nodes[newIdx].wall) {
                    if (atomicCAS((int*)&nodes[newIdx].visited, 0, 1) == 0) {
                        nodes[newIdx].parentIndex = nodeIdx;
                        int pos = atomicAdd(nextFrontierSize, 1);
                        nextFrontier[pos] = newIdx;
                        if (newIdx == endIdx) {
                            *levelCompleted = true;
                        }
                    }
                }
            }
        }
    }
}

// Funzione di inizializzazione dei nodi
void initializeNodes(char maze[][COLS], Node* nodes, Point* start, Point* end, MazeConfig config) {
    int nodeCount = 0;
    
    for (int i = 0; i < config.rows; i++) {
        for (int j = 0; j < config.cols; j++) {
            int idx = coordToIndex(i, j, &config);
            nodes[idx].pos = makePoint(i, j);
            nodes[idx].visited = false;
            nodes[idx].wall = (maze[i][j] == '#');
            nodes[idx].parentIndex = -1;
            
            if (maze[i][j] == '#') {
                nodes[idx].nodeNum = -1;
            } else {
                nodes[idx].nodeNum = nodeCount++;
            }
            
            if (maze[i][j] == '2') {
                *start = makePoint(i, j);
            } else if (maze[i][j] == '3') {
                *end = makePoint(i, j);
            }
        }
    }

    // Stampa il labirinto con i numeri dei nodi
    printf("\nLabirinto con nodi numerati:\n");
    for (int i = 0; i < config.rows; i++) {
        for (int j = 0; j < config.cols; j++) {
            if (nodes[coordToIndex(i, j, &config)].wall) {
                printf("## ");
            } else {
                printf("%2d ", nodes[coordToIndex(i, j, &config)].nodeNum);
            }
        }
        printf("\n");
    }
}

// Funzione per la gestione degli errori CUDA
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

bool solveMazeCuda(Node* hostNodes, Point start, Point end, int* path, int* pathLength, MazeConfig config) {
    Node* deviceNodes;
    int *deviceFrontier, *deviceNextFrontier;
    int *deviceFrontierSize, *deviceNextFrontierSize;
    bool *deviceLevelCompleted;
    
    // Allocazione memoria basata sulla configurazione
    cudaMalloc(&deviceNodes, config.maxNodes * sizeof(Node));
    cudaMalloc(&deviceFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceNextFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceFrontierSize, sizeof(int));
    cudaMalloc(&deviceNextFrontierSize, sizeof(int));
    cudaMalloc(&deviceLevelCompleted, sizeof(bool));
    
    // Copia i nodi sulla GPU
    cudaMemcpy(deviceNodes, hostNodes, config.maxNodes * sizeof(Node), cudaMemcpyHostToDevice);
    
    // Inizializzazione frontiera
    int startIdx = coordToIndex(start.row, start.col, &config);
    int initialFrontierSize = 1;
    cudaMemcpy(deviceFrontier, &startIdx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFrontierSize, &initialFrontierSize, sizeof(int), cudaMemcpyHostToDevice);
    
    int endIdx = coordToIndex(end.row, end.col, &config);
    bool pathFound = false;
    
    // Ottieni informazioni sulla GPU
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    // Loop principale BFS con dimensionamento ottimizzato della griglia
    while (true) {
        int hostFrontierSize;
        cudaMemcpy(&hostFrontierSize, deviceFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (hostFrontierSize == 0) break;
        
        // Reset per il prossimo livello
        int zero = 0;
        bool false_val = false;
        cudaMemcpy(deviceNextFrontierSize, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceLevelCompleted, &false_val, sizeof(bool), cudaMemcpyHostToDevice);
        
        // Calcolo ottimizzato della griglia
        int numThreadsNeeded = hostFrontierSize;
        int numBlocks = (numThreadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Assicura un minimo di blocchi per SM per massimizzare l'occupancy
        int minBlocksPerSM = deviceProp.maxThreadsPerMultiProcessor / BLOCK_SIZE;
        int optimalMinBlocks = deviceProp.multiProcessorCount * minBlocksPerSM;
        numBlocks = max(numBlocks, optimalMinBlocks);
        
        // Limita il numero di blocchi se necessario
        int maxBlocks = deviceProp.maxGridSize[0];
        numBlocks = min(numBlocks, maxBlocks);
        
        exploreLevel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
            deviceNodes,
            deviceFrontier,
            deviceNextFrontier,
            deviceFrontierSize,
            deviceNextFrontierSize,
            deviceLevelCompleted,
            endIdx,
            config
        );
        
        // Verifica completamento
        bool levelCompleted;
        cudaMemcpy(&levelCompleted, deviceLevelCompleted, sizeof(bool), cudaMemcpyDeviceToHost);
        
        if (levelCompleted) {
            pathFound = true;
            break;
        }
        
        // Scambia le frontiere
        int* temp = deviceFrontier;
        deviceFrontier = deviceNextFrontier;
        deviceNextFrontier = temp;
        cudaMemcpy(deviceFrontierSize, deviceNextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Se abbiamo trovato un percorso, ricostruiscilo
    if (pathFound) {
        // Copia i nodi aggiornati indietro all'host
        cudaMemcpy(hostNodes, deviceNodes, MAX_NODES * sizeof(Node), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        // Ricostruisci il percorso (questa parte rimane sequenziale)
        *pathLength = 0;
        int currentIdx = endIdx;
        while (currentIdx != startIdx) {
            path[(*pathLength)++] = hostNodes[currentIdx].nodeNum;
            currentIdx = hostNodes[currentIdx].parentIndex;
        }
        path[(*pathLength)++] = hostNodes[startIdx].nodeNum;
        
        // Inverti il percorso
        for (int i = 0; i < *pathLength / 2; i++) {
            int temp = path[i];
            path[i] = path[*pathLength - 1 - i];
            path[*pathLength - 1 - i] = temp;
        }
    }
    
    // Cleanup
    cudaFree(deviceNodes);
    cudaFree(deviceFrontier);
    cudaFree(deviceNextFrontier);
    cudaFree(deviceFrontierSize);
    cudaFree(deviceNextFrontierSize);
    cudaFree(deviceLevelCompleted);
    
    return pathFound;
}


// Utility per stampare il labirinto
void printMaze(char maze[ROWS][COLS]) {
    printf("\nLabirinto iniziale:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%c ", maze[i][j]);
        }
        printf("\n");
    }
}

int loadMazesFromFile(const char* filename, char mazes[][ROWS][COLS]) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s (errno: %d)\n", filename, errno);
        perror("Error details");
        return 0;
    }

    int mazeCount = 0;
    char line[256];  // Make buffer larger to safely read longer lines
    int currentRow = 0;

    while (fgets(line, sizeof(line), file) != NULL) {
        // Remove newline characters (both \r and \n)
        line[strcspn(line, "\r\n")] = 0;
        
        size_t lineLen = strlen(line);
        printf("After removing newline - length: %zu, content: %s\n", lineLen, line);
        
        // Skip empty lines between mazes
        if (lineLen == 0) {
            if (currentRow == ROWS) {
                mazeCount++;
                currentRow = 0;
            }
            continue;
        }

        // Verify line length
        if (lineLen != COLS) {
            printf("Error: Invalid line length in maze %d, row %d (expected %d, got %zu)\n", 
                   mazeCount + 1, currentRow + 1, COLS, lineLen);
            fclose(file);
            return 0;
        }

        // Check if we've reached the maximum number of mazes
        if (mazeCount >= MAX_MAZES) {
            printf("Warning: Maximum number of mazes reached (%d)\n", MAX_MAZES);
            break;
        }

        // Copy the line into the maze array
        for (int col = 0; col < COLS; col++) {
            mazes[mazeCount][currentRow][col] = line[col];
        }

        currentRow++;

        // If we've read all rows for current maze
        if (currentRow == ROWS) {
            mazeCount++;
            currentRow = 0;
        }
    }

    // Handle the last maze if it's complete
    if (currentRow == ROWS) {
        mazeCount++;
    } else if (currentRow != 0) {
        printf("Error: Incomplete maze at end of file (only %d rows read)\n", currentRow);
        fclose(file);
        return mazeCount;
    }

    fclose(file);
    return mazeCount;
}

int main() {
    // Configurazione del labirinto
    MazeConfig h_config;
    h_config.rows     = ROWS;
    h_config.cols     = COLS;
    h_config.maxNodes = h_config.rows * h_config.cols;

    // Allocazione host
    Node* h_nodes  = (Node*)malloc(h_config.maxNodes * sizeof(Node));
    char* hostMaze = (char*)malloc(h_config.maxNodes * sizeof(char));

    // Puntatori device
    Node*      d_nodes;
    int*       d_frontier;
    int*       d_nextFrontier;
    int*       d_frontierSize;
    int*       d_nextFrontierSize;
    bool*      d_levelCompleted;
    MazeConfig* d_config;

    // Allocazioni su GPU
    cudaMalloc(&d_nodes,             h_config.maxNodes * sizeof(Node));
    cudaMalloc(&d_frontier,         h_config.maxNodes * sizeof(int));
    cudaMalloc(&d_nextFrontier,     h_config.maxNodes * sizeof(int));
    cudaMalloc(&d_frontierSize,     sizeof(int));
    cudaMalloc(&d_nextFrontierSize, sizeof(int));
    cudaMalloc(&d_levelCompleted,   sizeof(bool));
    cudaMalloc(&d_config,           sizeof(MazeConfig));

    // Copia della configurazione sulla GPU
    cudaMemcpy(d_config, &h_config, sizeof(MazeConfig), cudaMemcpyHostToDevice);

    // Caricamento dei labirinti da file
    char mazes[MAX_MAZES][ROWS][COLS];
    const char* filename = "mazes.txt";
    int numMazes = loadMazesFromFile(filename, mazes);
    if (numMazes == 0) {
        fprintf(stderr, "No mazes loaded from file. Exiting...\n");
        return EXIT_FAILURE;
    }

    printf("\n\nVersione CUDA_3 (parallelizzazione ottimizzata per grandi labirinti): \n");

    for (int i = 0; i < numMazes; ++i) {
        printf("\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Point start, end;
        printMaze(maze);

        // Inizializzazione dei nodi in host
        initializeNodes(maze, h_nodes, &start, &end, &h_config);

        // Risoluzione CUDA
        int path[MAX_NODES];
        int pathLength = 0;
        bool found = solveMazeCuda(h_nodes, start, end, path, &pathLength, &h_config);

        if (found) {
            printf("\nPercorso più breve (sequenza di nodi): ");
            for (int j = 0; j < pathLength; ++j) {
                printf("%d", path[j]);
                if (j + 1 < pathLength) printf(", ");
            }
            printf("\n");
        } else {
            printf("\nNessun percorso trovato!\n");
        }
    }

    // Cleanup GPU
    cudaFree(d_nodes);
    cudaFree(d_frontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_frontierSize);
    cudaFree(d_nextFrontierSize);
    cudaFree(d_levelCompleted);
    cudaFree(d_config);

    // Cleanup host
    free(h_nodes);
    free(hostMaze);

    return 0;
}