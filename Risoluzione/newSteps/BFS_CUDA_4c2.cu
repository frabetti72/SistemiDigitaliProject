/// QUARTA implementazione di CUDA
/// versione con uint64_t: supporto a 100x100
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


__global__ void debugFrontier(int* frontier, int frontierSize) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("DEBUG: Frontier size = %d\n", frontierSize);
        for (int i = 0; i < frontierSize && i < 20; i++) { // Print first 20 nodes
            printf("%d ", frontier[i]);
        }
        printf("\n");
    }
}


__global__ void reconstructPathKernel(
    CompactNode* nodes,          // Array of all nodes in the maze
    int* path,                   // Output buffer for the path
    int* pathLength,             // Pointer to store the final path length
    int startIdx,                // Index of the start node
    int endIdx,                  // Index of the end node
    int maxLength                // Maximum allowed path length
) {
    printf("---------------------------------------------------------------------------------------------------------------INIZIO RICOSTRUZIONE\n");
    // Shared memory for path reconstruction within a block
    __shared__ int sharedPath[BLOCK_SIZE];
    __shared__ int sharedPathLength;
    
    // Initialize the shared path length to 0
    if (threadIdx.x == 0) {
        sharedPathLength = 0;
    }
    __syncthreads();
    
    // Thread 0 reconstructs the path backward from end to start
    if (threadIdx.x == 0) {
        int currentIdx = endIdx;
        int currentLength = 0;
        bool validPath = true;
        
        // Trace backwards from end node to start node
        while (currentIdx != startIdx && currentLength < maxLength) {
            // Check for invalid indices
            if (currentIdx < 0 || currentIdx >= maxLength) {
                validPath = false;
                break;
            }
            
            // Store the current node number in the path
            sharedPath[currentLength++] = getNodeNum(&nodes[currentIdx]);
            
            // Get parent index for the next iteration
            int parentIdx = getParentIndex(&nodes[currentIdx]);
            
            // Check for cycles or corrupted parent indices
            if (parentIdx == currentIdx || parentIdx < 0 || parentIdx >= maxLength) {
                validPath = false;
                break;
            }
            
            currentIdx = parentIdx;
        }
        
        // Complete the path with the start node if we've found a valid path
        if (validPath && currentLength < maxLength) {
            sharedPath[currentLength++] = getNodeNum(&nodes[startIdx]);
            sharedPathLength = currentLength;
        } else if (!validPath) {
            // If path is invalid, set length to 0
            sharedPathLength = 0;
            printf("Error in path reconstruction: Invalid path detected\n");
        } else {
            // Path is too long
            sharedPathLength = 0;
            printf("Error in path reconstruction: Path exceeds maximum length\n");
        }
    }
    
    // Wait until thread 0 completes the path reconstruction
    __syncthreads();
    
    // If we have a valid path, reverse it (parallelize this operation)
    if (sharedPathLength > 0) {
        int halfLength = sharedPathLength / 2;
        
        for (int i = threadIdx.x; i < halfLength; i += blockDim.x) {
            int temp = sharedPath[i];
            sharedPath[i] = sharedPath[sharedPathLength - 1 - i];
            sharedPath[sharedPathLength - 1 - i] = temp;
        }
    }
    
    // Wait for all threads to complete the reversal
    __syncthreads();
    
    // Copy from shared memory to global memory (parallelize this operation)
    if (threadIdx.x == 0) {
        *pathLength = sharedPathLength;
    }
    
    for (int i = threadIdx.x; i < sharedPathLength; i += blockDim.x) {
        path[i] = sharedPath[i];
    }
    
    // Add debug output (only thread 0 prints)
    if (threadIdx.x == 0 && sharedPathLength > 0) {
        printf("Path reconstructed with length %d\n", sharedPathLength);
        printf("Path start: %d -> %d -> ... -> %d\n", 
               sharedPath[0], 
               (sharedPathLength > 1) ? sharedPath[1] : -1,
               (sharedPathLength > 0) ? sharedPath[sharedPathLength-1] : -1);
    }
}


// Versione corretta del kernel exploreLevel
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
    
    // Carica la frontiera in shared memory
    if (tid < *frontierSize) {
        sharedFrontier[localId] = frontier[tid];
    }
    __syncthreads();
    
    // Debug: stampa la dimensione della frontiera corrente
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Frontier size: %d\n", *frontierSize);
    }
    
    // Termina i thread non necessari
    if (tid >= *frontierSize) return;
    
    int nodeIdx = sharedFrontier[localId];
    
    // Validazione dell'indice del nodo
    if (nodeIdx < 0 || nodeIdx >= config.maxNodes) {
        if (threadIdx.x == 0) {
            printf("Invalid node index: %d\n", nodeIdx);
        }
        return;
    }
    
    CompactNode* currentNode = &nodes[nodeIdx];
    
    // Validazione dei valori del nodo corrente
    if (getRow(currentNode) < 0 || getRow(currentNode) >= config.rows || 
        getCol(currentNode) < 0 || getCol(currentNode) >= config.cols) {
        if (threadIdx.x == 0) {
            printf("Invalid node coordinates: (%d,%d)\n", getRow(currentNode), getCol(currentNode));
        }
        return;
    }
    
    // Esplora tutte le direzioni
    #pragma unroll
    for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        int newRow = getRow(currentNode) + directions[dir].row;
        int newCol = getCol(currentNode) + directions[dir].col;
        
        if (isValid(newRow, newCol, &config)) {
            int newIdx = coordToIndex(newRow, newCol, &config);
            
            // Validazione dell'indice del nuovo nodo
            if (newIdx < 0 || newIdx >= config.maxNodes) continue;
            
            CompactNode* newNode = &nodes[newIdx];
            
            // Verifica che non sia un muro e non sia già visitato
            if (!newNode->wall && !newNode->visited) {
                // Accesso atomico a visited
                if (atomicCAS((int*)&newNode->visited, 0, 1) == 0) {
                    // Imposta il parent index
                    setParentIndex(newNode, nodeIdx);
                    
                    // Aggiungi alla prossima frontiera
                    int position = atomicAdd(nextFrontierSize, 1);
                    
                    // Verifica che non ci sia overflow nella frontiera
                    if (position < config.maxNodes) {
                        nextFrontier[position] = newIdx;
                        
                        // Verifica se abbiamo raggiunto la destinazione
                        if (newIdx == endIdx) {
                            // Imposta il flag di completamento a vero
                            *levelCompleted = true;
                            
                            // Debug: stampa quando troviamo la destinazione
                            printf("Destination found! End node: %d at position (%d,%d)\n", 
                                   newIdx, newRow, newCol);
                        }
                    } else {
                        // Avviso per overflow della frontiera
                        printf("Warning: Next frontier overflow (%d)\n", position);
                    }
                }
            }
        }
    }
}

/*
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
        
        int tid     = blockIdx.x * blockDim.x + threadIdx.x;
        int localId = threadIdx.x;
        
        // 1) Copia dei dati della frontiera in shared mem
        if (tid < *frontierSize) {
            sharedFrontier[localId] = frontier[tid];
        }
        __syncthreads();
        
        // 2) Stampa una sola volta (thread 0 del blocco 0) della frontiera
        if (blockIdx.x == 0 && localId == 0) {
            // stampa dimensione
            printf("[DEBUG] Frontier size: %d\n", *frontierSize);
            // stampa nodi
            printf("[DEBUG] Frontier nodes: ");
            for (int i = 0; i < *frontierSize; ++i) {
                printf("%d ", sharedFrontier[i]);
            }
            printf("\n");
        }
        __syncthreads();
        
        // 3) Seleziona i thread utili all’esplorazione
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
                    // marcatore atomico di visita
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
 */       

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

// Correzione del ciclo principale BFS
bool solveMazeCuda(CompactNode* hostNodes, Point start, Point end, int* path, int* pathLength, MazeConfig config) {
    CompactNode* deviceNodes;
    int *deviceFrontier, *deviceNextFrontier;
    int *deviceFrontierSize, *deviceNextFrontierSize;
    bool *deviceLevelCompleted;
    
    // Debug: stampa coordinate di inizio e fine
    printf("Start position: (%d,%d)\n", start.row, start.col);
    printf("End position: (%d,%d)\n", end.row, end.col);
    
    // Allocazione memoria sulla GPU
    cudaMalloc(&deviceNodes, config.maxNodes * sizeof(CompactNode));
    cudaMalloc(&deviceFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceNextFrontier, config.maxNodes * sizeof(int));
    cudaMalloc(&deviceFrontierSize, sizeof(int));
    cudaMalloc(&deviceNextFrontierSize, sizeof(int));
    cudaMalloc(&deviceLevelCompleted, sizeof(bool));
    cudaCheckError();
    
    // Copia i nodi sulla GPU
    cudaMemcpy(deviceNodes, hostNodes, config.maxNodes * sizeof(CompactNode), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Inizializzazione frontiera
    int startIdx = coordToIndex(start.row, start.col, &config);
    int initialFrontierSize = 1;
    int zero = 0;
    bool false_val = false;
    
    // Marca il nodo di partenza come visitato
    hostNodes[startIdx].visited = true;
    
    // Copia le informazioni iniziali sulla GPU
    cudaMemcpy(deviceNodes, hostNodes, config.maxNodes * sizeof(CompactNode), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFrontier, &startIdx, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceFrontierSize, &initialFrontierSize, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    int endIdx = coordToIndex(end.row, end.col, &config);
    bool pathFound = false;
    
    // Debug: stampa indici di inizio e fine
    printf("Start index: %d, End index: %d\n", startIdx, endIdx);
    
    // Ottieni informazioni sulla GPU
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    // Copie host delle dimensioni delle frontiere
    int hostFrontierSize = 1;
    int hostNextFrontierSize = 0;
    
    int maxIterations = config.rows * config.cols; // Prevenzione di loop infiniti
    int iterations = 0;
    
    // Loop principale BFS 
    while (hostFrontierSize > 0 && iterations < maxIterations) {
        iterations++;
        
        // Reset per il prossimo livello
        cudaMemcpy(deviceNextFrontierSize, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceLevelCompleted, &false_val, sizeof(bool), cudaMemcpyHostToDevice);
        cudaCheckError();
        
        // Calcolo ottimizzato della griglia
        int numThreadsNeeded = hostFrontierSize;
        int numBlocks = (numThreadsNeeded + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Non forzare un minimo di blocchi se non necessario
        if (numThreadsNeeded < BLOCK_SIZE) {
            numBlocks = 1;
        }
        
        // Limita il numero di blocchi se necessario
        int maxBlocks = deviceProp.maxGridSize[0];
        numBlocks = min(numBlocks, maxBlocks);
        
        // Debug: dimensione della frontiera corrente
        printf("Iteration %d: frontierSize = %d, blocks = %d\n", iterations, hostFrontierSize, numBlocks);
        
        // Lancia il kernel con dimensione appropriata della shared memory
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
        cudaCheckError();
        
        // Verifica completamento
        bool levelCompleted;
        cudaMemcpy(&levelCompleted, deviceLevelCompleted, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        if (levelCompleted) {
            pathFound = true;
            printf("Path found at iteration %d!\n", iterations);
            break;
        }
        
        // Scambia le frontiere
        int* temp = deviceFrontier;
        deviceFrontier = deviceNextFrontier;
        deviceNextFrontier = temp;
        
        // Copia da device a device per la dimensione della frontiera
        cudaMemcpy(deviceFrontierSize, deviceNextFrontierSize, sizeof(int), cudaMemcpyDeviceToDevice);
        cudaCheckError();
        
        // Aggiorna la copia host della dimensione della frontiera
        cudaMemcpy(&hostFrontierSize, deviceFrontierSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckError();
    }
    
    // Verifica se abbiamo raggiunto il limite massimo di iterazioni
    if (iterations >= maxIterations && !pathFound) {
        printf("Warning: Reached maximum number of iterations (%d) without finding a path.\n", maxIterations);
    }
    
    // Se abbiamo trovato un percorso, ricostruiscilo
    if (pathFound) {
        // Copia i nodi aggiornati indietro all'host
        cudaMemcpy(hostNodes, deviceNodes, config.maxNodes * sizeof(CompactNode), cudaMemcpyDeviceToHost);
        cudaCheckError();
        
        // Debug: mostra l'indice del nodo finale e del suo genitore
        printf("End node index: %d\n", endIdx);
        printf("End node parent: %d\n", getParentIndex(&hostNodes[endIdx]));
        
        // Ricostruisci il percorso in modo più robusto
        *pathLength = 0;
        int currentIdx = endIdx;
        int safetyCounter = 0;
        int maxPathLength = config.rows * config.cols;
        int prevIdx = -1;  // Per rilevare cicli
        
        // Array temporaneo per memorizzare il percorso al contrario
        int tempPath[MAX_NODES];
        int tempLength = 0;
        
        while (currentIdx != startIdx && safetyCounter < maxPathLength) {
            // Verifica validità dell'indice corrente
            if (currentIdx < 0 || currentIdx >= config.maxNodes) {
                printf("Error: Invalid node index (%d) detected during path reconstruction.\n", currentIdx);
                pathFound = false;
                break;
            }
            
            // Aggiungi il nodo corrente al percorso temporaneo
            tempPath[tempLength++] = getNodeNum(&hostNodes[currentIdx]);
            
            // Ottieni l'indice del genitore
            int parentIdx = getParentIndex(&hostNodes[currentIdx]);
            
            // Verifica cicli o parent index corrotti
            if (parentIdx < 0 || parentIdx >= config.maxNodes || parentIdx == currentIdx || parentIdx == prevIdx) {
                printf("Error: Invalid parent index (%d) detected during path reconstruction.\n", parentIdx);
                pathFound = false;
                break;
            }
            
            // Passa al nodo genitore
            prevIdx = currentIdx;
            currentIdx = parentIdx;
            safetyCounter++;
            
            // Debug per ogni passo della ricostruzione
            if (safetyCounter % 10 == 0) {
                printf("Path reconstruction step %d: node %d -> parent %d\n", 
                       safetyCounter, prevIdx, currentIdx);
            }
        }
        
        if (currentIdx == startIdx && pathFound) {
            // Aggiungi il nodo di partenza al percorso temporaneo
            tempPath[tempLength++] = getNodeNum(&hostNodes[startIdx]);
            
            // Copia e inverti il percorso nel risultato finale
            for (int i = 0; i < tempLength; i++) {
                path[i] = tempPath[tempLength - 1 - i];
            }
            *pathLength = tempLength;
            
            printf("Path successfully reconstructed with length %d\n", *pathLength);
        } else if (safetyCounter >= maxPathLength) {
            printf("Error: Path reconstruction exceeded maximum length (%d).\n", maxPathLength);
            pathFound = false;
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

    printf("\n\nVersione CUDA_4 (riduzione dimensione di Node, con variazione per labirinti 100x100): \n");

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