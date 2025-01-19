/// QUARTA implementazione di CUDA
/// riduzione della struttura node da 24 bytes a 8 bytes : 
/// Utilizzo di mask, e compressione del dato dentro un intero
/// visited e wall occupano 1 bit!!!!
/// 
/// versione con struttura modificata per rendere wall e visited facilmente accessibili
///
/// Ricostruzione del percorso introdotta in CUDA_2 momentaneamente rimossa in favore di quella sequenziale: potenziali errori con grandi dimensioni

#include <stdint.h>  //aggiunta per configurazione colab (senza il compiler di colab non riconosce uint64_t)
#include <stdio.h>
#include <cuda_runtime.h>
#include <errno.h>

#define ROWS 100
#define COLS 100
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4
#define BLOCK_SIZE 256
#define MAX_MAZES 3 


// Maschere per i bit positions (versione 64 bit)
const uint64_t ROW_MASK = 0x3FF;          // 10 bits -> max 1023
const uint64_t COL_MASK = 0x3FF;          // 10 bits -> max 1023
const uint64_t NODE_NUM_MASK = 0xFFFFF;   // 20 bits -> max 1,048,575
const uint64_t PARENT_MASK = 0xFFFFF;     // 20 bits -> max 1,048,575

// Bit shifts per la nuova struttura
const int COL_SHIFT = 10;                  // After row
const int NODE_NUM_SHIFT = 20;             // After col
const int PARENT_SHIFT = 40;               // After nodeNum

typedef struct {
    uint64_t data;     // 64 bits !!!!
    bool visited;      // Separated flag
    bool wall;         // Separated flag
} CompactNode;

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

// Costanti per le direzioni (da copiare in memoria costante GPU)
__constant__ Point directions[NUM_DIRECTIONS] = {
    {-1, 0},  // UP
    {1, 0},   // DOWN
    {0, -1},  // LEFT
    {0, 1}    // RIGHT
};

/*
typedef struct {
    Point pos;
    bool visited;
    int nodeNum;
    int parentIndex;
    bool wall;
} Node;
*/

// Helper functions aggiornate per la nuova struttura
__host__ __device__ inline void setRow(CompactNode* node, int row) {
    node->data = (node->data & ~ROW_MASK) | (static_cast<uint64_t>(row) & ROW_MASK);
}

__host__ __device__ inline void setCol(CompactNode* node, int col) {
    node->data = (node->data & ~(COL_MASK << COL_SHIFT)) | 
                 ((static_cast<uint64_t>(col) & COL_MASK) << COL_SHIFT);
}

__host__ __device__ inline void setNodeNum(CompactNode* node, int num) {
    node->data = (node->data & ~(NODE_NUM_MASK << NODE_NUM_SHIFT)) |
                 ((static_cast<uint64_t>(num) & NODE_NUM_MASK) << NODE_NUM_SHIFT);
}

__host__ __device__ inline void setParentIndex(CompactNode* node, int parent) {
    node->data = (node->data & ~(PARENT_MASK << PARENT_SHIFT)) |
                 ((static_cast<uint64_t>(parent) & PARENT_MASK) << PARENT_SHIFT);
}

// Getter functions aggiornate
__host__ __device__ inline int getRow(CompactNode* node) {
    return static_cast<int>(node->data & ROW_MASK);
}

__host__ __device__ inline int getCol(CompactNode* node) {
    return static_cast<int>((node->data >> COL_SHIFT) & COL_MASK);
}

__host__ __device__ inline int getNodeNum(CompactNode* node) {
    return static_cast<int>((node->data >> NODE_NUM_SHIFT) & NODE_NUM_MASK);
}

__host__ __device__ inline int getParentIndex(CompactNode* node) {
    return static_cast<int>((node->data >> PARENT_SHIFT) & PARENT_MASK);
}

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
    CompactNode* nodes,
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
            sharedPath[currentLength++] = getNodeNum(&(nodes[currentIdx]));
            currentIdx = getParentIndex(&(nodes[currentIdx]));
        }
        if (currentLength < maxLength) {
            sharedPath[currentLength++] = getNodeNum(&(nodes[startIdx]));
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

__global__ void exploreLevel(
    CompactNode* nodes,
    int* frontier,
    int* nextFrontier,
    int* frontierSize,
    int* nextFrontierSize,
    bool* levelCompleted,
    int endIdx,
    const MazeConfig config
) {
    extern __shared__ int sharedMem[];
    int* sharedFrontier = sharedMem;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;
    
    if (tid < *frontierSize) {
        sharedFrontier[localId] = frontier[tid];
    }
    __syncthreads();
    
    if (tid >= *frontierSize) return;
    
    int nodeIdx = sharedFrontier[localId];
    CompactNode* currentNode = &nodes[nodeIdx];
    
    #pragma unroll
    for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        int newRow = getRow(currentNode) + directions[dir].row;
        int newCol = getCol(currentNode) + directions[dir].col;
        
        if (isValid(newRow, newCol, &config)) {
            int newIdx = coordToIndex(newRow, newCol, &config);
            CompactNode* newNode = &nodes[newIdx];
            
            if (!newNode->visited && !newNode->wall) {
                // Atomic operation semplificata per visited
                if (atomicCAS((int*)&newNode->visited, 0, 1) == 0) {
                    setParentIndex(newNode, nodeIdx);
                    
                    int position = atomicAdd(nextFrontierSize, 1);
                    nextFrontier[position] = newIdx;
                    
                    if (newIdx == endIdx) {
                        *levelCompleted = true;
                    }
                }
            }
        }
    }
}

// Funzione di inizializzazione aggiornata
void initializeNodes(char maze[][COLS], CompactNode* nodes, Point* start, Point* end, MazeConfig config) {
    int nodeCount = 0;
    
    for (int i = 0; i < config.rows; i++) {
        for (int j = 0; j < config.cols; j++) {
            int idx = coordToIndex(i, j, &config);
            CompactNode* currentNode = &nodes[idx];
            
            // Inizializza tutti i campi a 0
            currentNode->data = 0;
            currentNode->visited = false;
            currentNode->wall = (maze[i][j] == '#');
            
            // Setta i vari campi
            setRow(currentNode, i);
            setCol(currentNode, j);
            setParentIndex(currentNode, -1);
            
            if (currentNode->wall) {
                setNodeNum(currentNode, -1);
            } else {
                setNodeNum(currentNode, nodeCount++);
            }
            
            if (maze[i][j] == '2') {
                start->row = i;
                start->col = j;
            } else if (maze[i][j] == '3') {
                end->row = i;
                end->col = j;
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
                printf("%2d ", getNodeNum(&nodes[coordToIndex(i, j, &config)]));
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

bool solveMazeCuda(CompactNode* hostNodes, Point start, Point end, int* path, int* pathLength, MazeConfig config) {
    CompactNode* deviceNodes;
    int *deviceFrontier, *deviceNextFrontier;
    int *deviceFrontierSize, *deviceNextFrontierSize;
    bool *deviceLevelCompleted;
    
    // Allocazione memoria basata sulla configurazione
    cudaMalloc(&deviceNodes, config.maxNodes * sizeof(CompactNode));
    cudaMalloc(&deviceFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceNextFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceFrontierSize, sizeof(int));
    cudaMalloc(&deviceNextFrontierSize, sizeof(int));
    cudaMalloc(&deviceLevelCompleted, sizeof(bool));
    
    // Copia i nodi sulla GPU
    cudaMemcpy(deviceNodes, hostNodes, config.maxNodes * sizeof(CompactNode), cudaMemcpyHostToDevice);
    
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
        cudaMemcpy(hostNodes, deviceNodes, MAX_NODES * sizeof(CompactNode), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        // Ricostruisci il percorso (questa parte rimane sequenziale)
        *pathLength = 0;
        int currentIdx = endIdx;
        while (currentIdx != startIdx) {
            path[(*pathLength)++] = getNodeNum(&(hostNodes[currentIdx]));
            currentIdx = getParentIndex(&(hostNodes[currentIdx]));
        }
        path[(*pathLength)++] = getNodeNum(&(hostNodes[startIdx]));
        
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
    char mazes[MAX_MAZES][ROWS][COLS];
    const char* filename = "mazes100.txt";
    
    int numMazes = loadMazesFromFile(filename, mazes);
    if (numMazes == 0) {
        printf("No mazes loaded from file. Exiting...\n");
        return 1;
    }

    printf("\n\nVersione CUDA_4 (riduzione dimensione di Node, con variazione per prestazioni): \n");

    // Configura le dimensioni del labirinto
    MazeConfig config;
    config.rows = ROWS;  // Usa le dimensioni definite
    config.cols = COLS;
    config.maxNodes = config.rows * config.cols;

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Point start, end;
        CompactNode nodes[MAX_NODES];
        
        printMaze(maze);
        initializeNodes(maze, nodes, &start, &end, config);

        int path[MAX_NODES];
        int pathLength;

        if (solveMazeCuda(nodes, start, end, path, &pathLength, config)) {
            printf("\nPercorso più breve (sequenza di nodi): ");
            for (int j = 0; j < pathLength; j++) {
                printf("%d", path[j]);
                if (j < pathLength - 1) printf(", ");
            }
            printf("\n");
        } else {
            printf("\nNessun percorso trovato!\n");
        }
    }
    
    return 0;
}