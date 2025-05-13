/// Prima implementazione di CUDA
/// ogni thread si occupa di un nodo della frontiera (analizza i vicini e costruisce nuova frontiera)
/// exploreLevel è il kernel principale (analizza un livello alla volta => una chiamata per iterazione del livello)
/// 

#include <stdio.h>
#include <cuda_runtime.h>
#include <errno.h>

#define ROWS 50
#define COLS 50
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4
#define BLOCK_SIZE 256
#define MAX_MAZES 100

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

// Kernel principale per l'esplorazione BFS
__global__ void exploreLevel(
    Node* nodes,
    int* frontier,
    int* nextFrontier,
    int* frontierSize,
    int* nextFrontierSize,
    bool* levelCompleted,
    int endIdx
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= *frontierSize) return;
    
    int nodeIdx = frontier[tid];
    Node* currentNode = &nodes[nodeIdx];
    
    // Esplora tutte le direzioni
    for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        Point newPos = {
            currentNode->pos.row + directions[dir].row,
            currentNode->pos.col + directions[dir].col
        };
        
        if (isValid(newPos.row, newPos.col)) {
            int newIdx = coordToIndex(newPos.row, newPos.col);
            
            // Se troviamo un nodo non visitato e non muro
            if (!nodes[newIdx].visited && !nodes[newIdx].wall) {
                nodes[newIdx].visited = true;
                nodes[newIdx].parentIndex = nodeIdx;
                
                // Aggiungi alla nuova frontiera atomicamente
                int position = atomicAdd(nextFrontierSize, 1);
                nextFrontier[position] = newIdx;
                
                // Se abbiamo trovato l'uscita
                if (newIdx == endIdx) {
                    *levelCompleted = true;
                }
            }
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

    // Stampa il labirinto con i numeri dei nodi
    printf("\nLabirinto con nodi numerati:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (nodes[coordToIndex(i, j)].wall) {
                printf("## ");
            } else {
                printf("%2d ", nodes[coordToIndex(i, j)].nodeNum);
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

// Funzione host per inizializzare e gestire la risoluzione
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
    
    // Loop principale BFS
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
        
        // Lancia il kernel
        int numBlocks = (hostFrontierSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
    const char* filename = "mazes.txt";  // Your input file name
    
    int numMazes = loadMazesFromFile(filename, mazes);
    if (numMazes == 0) {
        printf("No mazes loaded from file. Exiting...\n");
        return 1;
    }

    printf("\n\nVersione CUDA_1 (parallelizzazione esplorazione nodi): \n");

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