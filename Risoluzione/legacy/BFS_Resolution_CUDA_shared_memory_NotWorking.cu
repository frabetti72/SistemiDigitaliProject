/// Prima implementazione di CUDA
/// ogni thread si occupa di un nodo della frontiera (analizza i vicini e costruisce nuova frontiera)
/// exploreLevel è il kernel principale (analizza un livello alla volta => una chiamata per iterazione del livello)
/// 

#include <stdio.h>
#include <cuda_runtime.h>

#define ROWS 6
#define COLS 7
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4
#define BLOCK_SIZE 256

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
__host__ __device__ bool isValid(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

__host__ __device__ int coordToIndex(int row, int col) {
    return row * COLS + col;
}

__host__ __device__ Point makePoint(int row, int col) {
    Point p;
    p.row = row;
    p.col = col;
    return p;
}

//nuovo kernel (much wow)
__global__ void exploreLevel(
    Node* nodes,
    int* frontier,
    int* nextFrontier,
    int* frontierSize,
    int* nextFrontierSize,  // Rimosso volatile
    bool* levelCompleted,
    int endIdx
) {
    __shared__ int sharedFrontier[BLOCK_SIZE];
    __shared__ int sharedNextSize;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    
    // Inizializza la shared memory
    if (lid == 0) {
        sharedNextSize = 0;
    }
    __syncthreads();
    
    // Carica il nodo della frontiera in shared memory
    int nodeIdx = -1;
    if (tid < *frontierSize) {
        nodeIdx = frontier[tid];
        sharedFrontier[lid] = nodeIdx;
    }
    __syncthreads();
    
    if (nodeIdx != -1) {
        Node* currentNode = &nodes[nodeIdx];
        int localNextNodes[NUM_DIRECTIONS];
        int localNextCount = 0;
        
        // Esplora tutte le direzioni
        for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
            Point newPos = {
                currentNode->pos.row + directions[dir].row,
                currentNode->pos.col + directions[dir].col
            };
            
            if (isValid(newPos.row, newPos.col)) {
                int newIdx = coordToIndex(newPos.row, newPos.col);
                
                if (!nodes[newIdx].visited && !nodes[newIdx].wall) {
                    // Usa atomicCAS invece di atomicExch per il flag visited
                    if (atomicCAS((unsigned int*)&nodes[newIdx].visited, false, true) == false) {
                        nodes[newIdx].parentIndex = nodeIdx;
                        localNextNodes[localNextCount++] = newIdx;
                        
                        if (newIdx == endIdx) {
                            *levelCompleted = true;
                        }
                    }
                }
            }
        }
        
        // Aggiungi i nodi locali alla shared memory
        if (localNextCount > 0) {
            int localBase = atomicAdd((int*)&sharedNextSize, localNextCount);
            
            if (localBase + localNextCount < BLOCK_SIZE) {
                for (int i = 0; i < localNextCount; i++) {
                    sharedFrontier[localBase + i] = localNextNodes[i];
                }
            }
        }
    }
    __syncthreads();
    
    // Copia i risultati in memoria globale
    if (lid == 0 && sharedNextSize > 0) {
        int globalBase = atomicAdd(nextFrontierSize, sharedNextSize);
        for (int i = 0; i < sharedNextSize; i++) {
            nextFrontier[globalBase + i] = sharedFrontier[i];
        }
    }
}


// Funzione di inizializzazione dei nodi
void initializeNodes(char maze[ROWS][COLS], Node* nodes, Point* start, Point* end) {
    int nodeCount = 0;
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int idx = coordToIndex(i, j);
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
}

// Funzione per la gestione degli errori CUDA
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}


// Modifica la funzione solveMazeCuda per gestire la nuova frontiera
bool solveMazeCuda(Node* hostNodes, Point start, Point end, int* path, int* pathLength) {
    Node* deviceNodes;
    int *deviceFrontier, *deviceNextFrontier;
    int *deviceFrontierSize, *deviceNextFrontierSize;
    bool *deviceLevelCompleted;
    
    // Alloca memoria sulla GPU
    cudaMalloc(&deviceNodes, MAX_NODES * sizeof(Node));
    cudaMalloc(&deviceFrontier, MAX_NODES * sizeof(int));
    cudaMalloc(&deviceNextFrontier, MAX_NODES * sizeof(int));
    cudaMalloc(&deviceFrontierSize, sizeof(int));
    cudaMalloc(&deviceNextFrontierSize, sizeof(int));
    cudaMalloc(&deviceLevelCompleted, sizeof(bool));
    cudaCheckError();
    
    // Copia i nodi sulla GPU
    cudaMemcpy(deviceNodes, hostNodes, MAX_NODES * sizeof(Node), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Inizializza la frontiera con il nodo di partenza
    int startIdx = coordToIndex(start.row, start.col);
    int initialFrontierSize = 1;
    cudaMemcpy(deviceFrontier, &startIdx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFrontierSize, &initialFrontierSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    int endIdx = coordToIndex(end.row, end.col);
    bool pathFound = false;
    
    int* compactFrontier;
    cudaMalloc(&compactFrontier, MAX_NODES * sizeof(int));
    cudaCheckError();
    
    while (true) {
        int hostFrontierSize;
        cudaMemcpy(&hostFrontierSize, deviceFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        if (hostFrontierSize == 0) break;
        
        // Reset per il prossimo livello
        int zero = 0;
        bool false_val = false;
        cudaMemcpy(deviceNextFrontierSize, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceLevelCompleted, &false_val, sizeof(bool), cudaMemcpyHostToDevice);
        cudaCheckError();
        
        // Calcola la griglia ottimale
        int numBlocks = (hostFrontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Lancia il kernel ottimizzato
        exploreLevel<<<numBlocks, BLOCK_SIZE>>>(
            deviceNodes,
            deviceFrontier,
            deviceNextFrontier,
            deviceFrontierSize,
            deviceNextFrontierSize,
            deviceLevelCompleted,
            endIdx
        );
        cudaCheckError();
        
        // Controlla se abbiamo trovato il percorso
        bool levelCompleted;
        cudaMemcpy(&levelCompleted, deviceLevelCompleted, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        if (levelCompleted) {
            pathFound = true;
            break;
        }
        
        // Scambia le frontiere
        int* temp = deviceFrontier;
        deviceFrontier = deviceNextFrontier;
        deviceNextFrontier = temp;
        cudaMemcpy(deviceFrontierSize, deviceNextFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckError();
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
    
    // Libera la memoria
    cudaFree(deviceNodes);
    cudaFree(deviceFrontier);
    cudaFree(deviceNextFrontier);
    cudaFree(deviceFrontierSize);
    cudaFree(deviceNextFrontierSize);
    cudaFree(deviceLevelCompleted);
    cudaFree(compactFrontier); //aggiunto

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

int main() {
    char mazes[][ROWS][COLS] = {
        {
            {'0','0','#','0','0','0','0'},
            {'0','2','#','0','0','0','0'},
            {'0','0','#','0','0','0','0'},
            {'0','0','0','#','0','3','0'},
            {'0','#','#','#','#','0','0'},
            {'0','0','0','0','0','0','0'}
        },
        {
            {'0','0','0','0','0','0','0'},
            {'0','2','#','#','#','#','0'},
            {'0','0','0','0','0','#','0'},
            {'0','#','#','#','0','#','0'},
            {'0','0','0','0','0','3','0'},
            {'0','#','#','#','#','0','0'}
        },
        {
            {'0','0','0','0','0','#','0'},
            {'0','2','#','#','#','#','0'},
            {'0','0','0','0','0','#','0'},
            {'0','#','#','#','0','#','0'},
            {'0','0','0','0','#','3','0'},
            {'0','#','#','#','#','0','0'}
        }
    };

    int numMazes = sizeof(mazes) / sizeof(mazes[0]);

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Point start, end;
        Node nodes[MAX_NODES];
        
        printMaze(maze);
        initializeNodes(maze, nodes, &start, &end);

        int path[MAX_NODES];
        int pathLength;

        if (solveMazeCuda(nodes, start, end, path, &pathLength)) {
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