/// versione ristrutturata per CUDA, lo scopo è creare una struttura che ora risulti parallelizzabile
///

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h> 
#include <errno.h>

#define ROWS 50
#define COLS 50
#define MAX_NODES (ROWS * COLS)
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



//controlla se il nodo è in una posizione valida
bool isValid(int x, int y, int rows, int cols) {
    return x >= 0 && x < rows && y >= 0 && y < cols; 
}

//i nodi sono numerati come le celle
// ci sono dei numeri senza nodo corrispondenti ai wall
int coordToIndex(int x, int y, int cols) {
    return x * cols + y;
}


// Simula l'esplorazione di un singolo nodo (equivalente a un thread in CUDA)
void exploreNode(
    Node* nodes,
    int nodeIdx,
    int* nextFrontier,
    int* nextFrontierSize,
    bool* levelCompleted,
    int endIdx
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
        if (isValid(newX, newY, ROWS, COLS)) {
            int neighborIdx = coordToIndex(newX, newY, COLS);
            
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
void printMaze(int rows, int cols, char maze[][COLS])
{
    puts("\nLabirinto iniziale:");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            putchar(maze[i][j]);
            putchar(' ');
        }
        putchar('\n');
    }
}

//stampa il labirinto con numeri
void printMazeNumbered(int rows, int cols, char maze[][COLS])
{
    puts("\nLabirinto numerato:");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int nodeId = coordToIndex(i, j, cols);
            if (maze[i][j] == '#') {
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
int initializeNodes(int rows, int cols, char maze[][COLS], Node nodes[]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = coordToIndex(i, j,cols);

            nodes[idx].nodeId=idx;
            nodes[idx].visited= false;
            nodes[idx].parentIndex = NOT_VISITED;
            nodes[idx].x=i;
            nodes[idx].y=j;
            nodes[idx].wall  = (maze[i][j] == '#');
            nodes[idx].start = (maze[i][j] == '2');
            nodes[idx].goal  = (maze[i][j] == '3');

        }
    }
    printMazeNumbered(rows,cols,maze);
    return rows * cols; // Aggiungo il return mancante
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


// Funzione principale per la risoluzione del labirinto
bool solveMazeIntermediate(int maxNodes, Node* nodes, int* path, int* pathLength) {
    int* frontier = (int*)malloc(MAX_NODES * sizeof(int));
    int* nextFrontier = (int*)malloc(MAX_NODES * sizeof(int));
    int frontierSize = 0;
    int nextFrontierSize = 0;
    bool levelCompleted = false;
    
    int startx, starty, goalx, goaly;
    bool flags=false;
    bool flagg=false;
    for(int i = 0; i<maxNodes ; i++ ){
        if(nodes[i].goal){
            goalx=nodes[i].x;
            goaly=nodes[i].y;
            flagg=true;
        }
        if(nodes[i].start){
            startx=nodes[i].x;
            starty=nodes[i].y;
            flags=true;
        }
    }
    if(!(flags&&flagg)){
        puts("\nflag non rispettate, labirinto mal formato");
        free(frontier);
        free(nextFrontier);
        return false;
    }
    // Inizializza la frontiera con il nodo di partenza
    int startIdx = coordToIndex(startx, starty,COLS); //todo sostituire cols
    frontier[frontierSize++] = startIdx;
    nodes[startIdx].visited = true;
    
    int endIdx = coordToIndex(goalx,goaly,COLS); //todo sostituire cols
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
                endIdx
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
    int rows = ROWS;
    int cols = COLS;
    char mazes[MAX_MAZES][ROWS][COLS];
    const char* filename = "mazes.txt";  // Your input file name
    
    int numMazes = loadMazesFromFile(filename, mazes);
    if (numMazes == 0) {
        printf("No mazes loaded from file. Exiting...\n");
        return 1;
    }

    printf("\n\nVersione sequenziale: \n");

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Node nodes[MAX_NODES];
        
        printMaze(rows, cols, maze);
        int nodeCount = initializeNodes(rows, cols, maze, nodes);
        

        int path[MAX_NODES];
        int pathLength;

        if (solveMazeIntermediate(nodeCount, nodes, path, &pathLength)) { 
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