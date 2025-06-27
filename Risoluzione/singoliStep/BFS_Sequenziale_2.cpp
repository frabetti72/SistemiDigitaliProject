/// versione ristrutturata per CUDA, lo scopo è creare una struttura che ora risulti parallelizzabile
/// Versione con dimensioni dinamiche, no COLS e ROWS
/// testati con multipli 50x50

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> 
#include <errno.h>

#define NUM_DIRECTIONS 4
#define MAX_MAZES 100 
#define NOT_VISITED -1

typedef struct {
    int x; // x del nodo 
    int y;  // y del nodo
    bool visited; // true if visited
    int nodeId; // id del nodo   
    int parentIndex; // chi l'ha vistato (-1 se non visitato)
    bool wall; //true if wall
    bool start; //dice se è start
    bool goal; //dice se è il goal
} Node;

typedef struct {
    char** data;
    int rows;
    int cols;
} Maze;

//controlla se il nodo è in una posizione valida
bool isValid(int x, int y, int rows, int cols) {
    return x >= 0 && x < rows && y >= 0 && y < cols; 
}

//i nodi sono numerati come le celle
// ci sono dei numeri senza nodo corrispondenti ai wall
int coordToIndex(int x, int y, int cols) {
    return x * cols + y;
}

// Alloca memoria per un labirinto
Maze* createMaze(int rows, int cols) {
    Maze* maze = (Maze*)malloc(sizeof(Maze));
    if (!maze) return NULL;
    
    maze->rows = rows;
    maze->cols = cols;
    maze->data = (char**)malloc(rows * sizeof(char*));
    if (!maze->data) {
        free(maze);
        return NULL;
    }
    
    for (int i = 0; i < rows; i++) {
        maze->data[i] = (char*)malloc(cols * sizeof(char));
        if (!maze->data[i]) {
            // Cleanup in caso di errore
            for (int j = 0; j < i; j++) {
                free(maze->data[j]);
            }
            free(maze->data);
            free(maze);
            return NULL;
        }
    }
    
    return maze;
}

// Libera memoria del labirinto
void freeMaze(Maze* maze) {
    if (!maze) return;
    
    for (int i = 0; i < maze->rows; i++) {
        free(maze->data[i]);
    }
    free(maze->data);
    free(maze);
}

// Simula l'esplorazione di un singolo nodo (equivalente a un thread in CUDA)
void exploreNode(
    Node* nodes,
    int nodeIdx,
    int* nextFrontier,
    int* nextFrontierSize,
    bool* levelCompleted,
    int endIdx,
    int rows,
    int cols
) {
    if (nodeIdx == endIdx) {
        *levelCompleted = true;
        return;
    }
    
    // Le 4 direzioni: su, giù, sinistra, destra
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    int currentX = nodes[nodeIdx].x;
    int currentY = nodes[nodeIdx].y;
    
    // Esplora i 4 vicini
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
        int newX = currentX + dx[i];
        int newY = currentY + dy[i];
        
        // Controlla se la posizione è valida
        if (isValid(newX, newY, rows, cols)) {
            int neighborIdx = coordToIndex(newX, newY, cols);
            
            // Se il vicino non è un muro, non è stato visitato
            if (!nodes[neighborIdx].wall && !nodes[neighborIdx].visited) {
                nodes[neighborIdx].visited = true;
                nodes[neighborIdx].parentIndex = nodeIdx;
                nextFrontier[(*nextFrontierSize)++] = neighborIdx;
                
                // Se abbiamo raggiunto il goal
                if (neighborIdx == endIdx) {
                    *levelCompleted = true;
                }
            }
        }
    }
}

//stampa il labirinto 
void printMaze(Maze* maze)
{
    puts("\nLabirinto iniziale:");
    for (int i = 0; i < maze->rows; ++i) {
        for (int j = 0; j < maze->cols; ++j) {
            putchar(maze->data[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

//stampa il labirinto con numeri
void printMazeNumbered(Maze* maze)
{
    puts("\nLabirinto numerato:");
    for (int i = 0; i < maze->rows; ++i) {
        for (int j = 0; j < maze->cols; ++j) {
            int nodeId = coordToIndex(i, j, maze->cols);
            if (maze->data[i][j] == '#') {
                printf("### ");
            } else {
                printf("%3d ", nodeId);
            }
        }
        putchar('\n');
    }
}

// partiamo con nodes che è un insieme di nodi vuoti con dimensioni pari al labirinto
// li inizializziamo
int initializeNodes(Maze* maze, Node* nodes) {
    int maxNodes = maze->rows * maze->cols;
    
    for (int i = 0; i < maze->rows; i++) {
        for (int j = 0; j < maze->cols; j++) {
            int idx = coordToIndex(i, j, maze->cols);

            nodes[idx].nodeId = idx;
            nodes[idx].visited = false;
            nodes[idx].parentIndex = NOT_VISITED;
            nodes[idx].x = i;
            nodes[idx].y = j;
            nodes[idx].wall = (maze->data[i][j] == '#');
            nodes[idx].start = (maze->data[i][j] == '2');
            nodes[idx].goal = (maze->data[i][j] == '3');
        }
    }
    printMazeNumbered(maze);
    return maxNodes;
}

int loadMazesFromFile(const char* filename, Maze*** mazes) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s (errno: %d)\n", filename, errno);
        perror("Error details");
        return 0;
    }

    *mazes = (Maze**)malloc(MAX_MAZES * sizeof(Maze*));
    if (!*mazes) {
        fclose(file);
        return 0;
    }

    int mazeCount = 0;
    char line[1024];
    char tempLines[1000][1024]; // Buffer temporaneo per le righe
    int tempRowCount = 0;

    printf("Starting to read mazes from file...\n");

    while (fgets(line, sizeof(line), file) != NULL && mazeCount < MAX_MAZES) {
        // Remove newline characters
        line[strcspn(line, "\r\n")] = 0;
        
        size_t lineLen = strlen(line);
        printf("Read line %d: length=%zu, content='%s'\n", tempRowCount, lineLen, line);
        
        // Se la riga è vuota, concludi il labirinto corrente
        if (lineLen == 0) {
            if (tempRowCount > 0) {
                // Crea il labirinto con le righe accumulate
                int cols = strlen(tempLines[0]);
                (*mazes)[mazeCount] = createMaze(tempRowCount, cols);
                
                if ((*mazes)[mazeCount]) {
                    printf("Creating maze %d: %dx%d\n", mazeCount + 1, tempRowCount, cols);
                    for (int i = 0; i < tempRowCount; i++) {
                        strcpy((*mazes)[mazeCount]->data[i], tempLines[i]);
                    }
                    mazeCount++;
                }
                tempRowCount = 0;
            }
            continue;
        }

        // Aggiungi la riga al buffer temporaneo
        if (tempRowCount < 1000) {
            strcpy(tempLines[tempRowCount], line);
            tempRowCount++;
        }
    }

    // Gestisci l'ultimo labirinto se non c'è una riga vuota finale
    if (tempRowCount > 0) {
        int cols = strlen(tempLines[0]);
        (*mazes)[mazeCount] = createMaze(tempRowCount, cols);
        
        if ((*mazes)[mazeCount]) {
            printf("Creating final maze %d: %dx%d\n", mazeCount + 1, tempRowCount, cols);
            for (int i = 0; i < tempRowCount; i++) {
                strcpy((*mazes)[mazeCount]->data[i], tempLines[i]);
            }
            mazeCount++;
        }
    }

    fclose(file);
    printf("Total mazes loaded: %d\n", mazeCount);
    return mazeCount;
}

// Funzione principale per la risoluzione del labirinto
bool solveMazeIntermediate(Maze* maze, Node* nodes, int* path, int* pathLength) {
    int maxNodes = maze->rows * maze->cols;
    int* frontier = (int*)malloc(maxNodes * sizeof(int));
    int* nextFrontier = (int*)malloc(maxNodes * sizeof(int));
    int frontierSize = 0;
    int nextFrontierSize = 0;
    bool levelCompleted = false;
    
    int startx, starty, goalx, goaly;
    bool flags = false;
    bool flagg = false;
    
    for(int i = 0; i < maxNodes; i++){
        if(nodes[i].goal){
            goalx = nodes[i].x;
            goaly = nodes[i].y;
            flagg = true;
        }
        if(nodes[i].start){
            startx = nodes[i].x;
            starty = nodes[i].y;
            flags = true;
        }
    }
    
    if(!(flags && flagg)){
        puts("\nflag non rispettate, labirinto mal formato");
        free(frontier);
        free(nextFrontier);
        return false;
    }
    
    // Inizializza la frontiera con il nodo di partenza
    int startIdx = coordToIndex(startx, starty, maze->cols);
    frontier[frontierSize++] = startIdx;
    nodes[startIdx].visited = true;
    
    int endIdx = coordToIndex(goalx, goaly, maze->cols);
    bool pathFound = false;
    
    // Loop principale BFS
    while (frontierSize > 0) {
        nextFrontierSize = 0;
        levelCompleted = false;
        
        // Esplora tutti i nodi nella frontiera corrente
        for (int i = 0; i < frontierSize; i++) {
            exploreNode(
                nodes,
                frontier[i],
                nextFrontier,
                &nextFrontierSize,
                &levelCompleted,
                endIdx,
                maze->rows,
                maze->cols
            );
        }
        
        // Se abbiamo trovato il percorso
        if (levelCompleted) {
            pathFound = true;
            break;
        }
        
        // Scambia le frontiere
        int* temp = frontier;
        frontier = nextFrontier;
        nextFrontier = temp;
        frontierSize = nextFrontierSize;
    }
    
    // Se abbiamo trovato un percorso, ricostruiscilo
    if (pathFound) {
        *pathLength = 0;
        int currentIdx = endIdx;
        
        // Ricostruisci il percorso dall'end al start
        while (currentIdx != startIdx) {
            path[(*pathLength)++] = nodes[currentIdx].nodeId;
            currentIdx = nodes[currentIdx].parentIndex;
        }
        path[(*pathLength)++] = nodes[startIdx].nodeId;
        
        // Inverti il percorso
        for (int i = 0; i < *pathLength / 2; i++) {
            int temp = path[i];
            path[i] = path[*pathLength - 1 - i];
            path[*pathLength - 1 - i] = temp;
        }
    }
    
    // Libera la memoria
    free(frontier);
    free(nextFrontier);
    
    return pathFound;
}

int main() {
    Maze** mazes;
    const char* filename = "mazes.txt";
    
    int numMazes = loadMazesFromFile(filename, &mazes);
    if (numMazes == 0) {
        printf("No mazes loaded from file. Exiting...\n");
        return 1;
    }

    printf("\n\nVersione sequenziale con dimensioni dinamiche: \n");

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nLabirinto %d (%dx%d):\n", i + 1, mazes[i]->rows, mazes[i]->cols);
        
        int maxNodes = mazes[i]->rows * mazes[i]->cols;
        Node* nodes = (Node*)malloc(maxNodes * sizeof(Node));
        
        if (!nodes) {
            printf("Errore allocazione memoria per i nodi\n");
            continue;
        }
        
        printMaze(mazes[i]);
        int nodeCount = initializeNodes(mazes[i], nodes);
        
        int* path = (int*)malloc(maxNodes * sizeof(int));
        if (!path) {
            printf("Errore allocazione memoria per il percorso\n");
            free(nodes);
            continue;
        }
        
        int pathLength;

        if (solveMazeIntermediate(mazes[i], nodes, path, &pathLength)) { 
            printf("\nPercorso più breve (sequenza di nodi): ");
            for (int j = 0; j < pathLength; j++) {
                printf("%d", path[j]);
                if (j < pathLength - 1) printf(", ");
            }
            printf("\n");
        } else {
            printf("\nNessun percorso trovato!\n");
        }
        
        free(nodes);
        free(path);
    }
    
    // Libera memoria dei labirinti
    for (int i = 0; i < numMazes; i++) {
        freeMaze(mazes[i]);
    }
    free(mazes);
    
    return 0;
}