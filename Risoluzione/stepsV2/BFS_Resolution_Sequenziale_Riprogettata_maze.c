/// versione ristrutturata per CUDA, lo scopo è creare una struttura che ora risulti parallelizzabile
///

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ROWS 20
#define COLS 20
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4

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

// Direzioni come array globale (simile a __constant__ in CUDA)
const Point directions[NUM_DIRECTIONS] = {
    {-1, 0},  // UP
    {1, 0},   // DOWN
    {0, -1},  // LEFT
    {0, 1}    // RIGHT
};

bool isValid(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

int coordToIndex(int row, int col) {
    return row * COLS + col;
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
                
                // Aggiungi alla nuova frontiera
                nextFrontier[*nextFrontierSize] = newIdx;
                (*nextFrontierSize)++;
                
                // Se abbiamo trovato l'uscita
                if (newIdx == endIdx) {
                    *levelCompleted = true;
                }
            }
        }
    }
}

// Funzione principale per la risoluzione del labirinto
bool solveMazeIntermediate(Node* nodes, Point start, Point end, int* path, int* pathLength) {
    int* frontier = (int*)malloc(MAX_NODES * sizeof(int));
    int* nextFrontier = (int*)malloc(MAX_NODES * sizeof(int));
    int frontierSize = 0;
    int nextFrontierSize = 0;
    bool levelCompleted = false;
    
    // Inizializza la frontiera con il nodo di partenza
    int startIdx = coordToIndex(start.row, start.col);
    frontier[frontierSize++] = startIdx;
    nodes[startIdx].visited = true;
    
    int endIdx = coordToIndex(end.row, end.col);
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
            path[(*pathLength)++] = nodes[currentIdx].nodeNum;
            currentIdx = nodes[currentIdx].parentIndex;
        }
        path[(*pathLength)++] = nodes[startIdx].nodeNum;
        
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

void printMaze(char maze[ROWS][COLS]) {
    printf("\nLabirinto iniziale:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%c ", maze[i][j]);
        }
        printf("\n");
    }
}

// Funzione per inizializzare i nodi dal labirinto
int initializeNodes(char maze[ROWS][COLS], Node nodes[], Point* start, Point* end) {
    int nodeCount = 0;
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int idx = coordToIndex(i, j);
            nodes[idx].pos = (Point){i, j};
            nodes[idx].visited = false;
            nodes[idx].wall = (maze[i][j] == '#');
            nodes[idx].parentIndex = -1;
            
            if (maze[i][j] == '#') {
                nodes[idx].nodeNum = -1;
            } else {
                nodes[idx].nodeNum = nodeCount++;
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
    
    return nodeCount;
}

int main() {
    char mazes[][ROWS][COLS] = {
        {
            {'2','#','0','#','0','0','0','0','0','0','0','#','0','0','0','0','0','#','0','0'},
            {'0','#','0','#','0','#','#','0','#','#','0','#','0','#','0','#','0','0','#','0'},
            {'0','#','0','0','0','#','0','0','0','#','0','0','#','0','0','0','#','0','#','0'},
            {'0','#','0','#','#','0','0','#','#','0','0','#','0','0','#','0','#','0','0','0'},
            {'0','0','#','0','#','0','#','#','0','0','#','0','0','#','#','#','0','#','#','0'},
            {'#','0','0','0','0','0','#','0','0','#','#','#','0','0','#','0','0','#','0','0'},
            {'0','#','#','#','0','#','#','0','#','0','#','0','#','0','#','0','#','0','0','#'},
            {'0','0','0','0','#','0','0','0','#','0','0','0','#','0','#','0','#','#','0','#'},
            {'0','#','#','0','0','#','0','#','0','#','#','0','0','0','#','0','0','0','0','0'},
            {'0','#','#','#','#','#','0','0','0','0','0','0','#','#','0','0','#','#','#','0'},
            {'0','0','#','0','0','0','#','#','#','#','#','#','0','#','0','#','0','0','0','#'},
            {'#','0','#','0','#','0','0','0','0','0','0','#','0','0','0','#','0','#','0','0'},
            {'#','0','0','0','0','#','#','0','#','#','0','#','0','#','#','0','0','#','#','#'},
            {'0','0','#','#','#','#','0','0','#','0','0','#','0','0','0','#','0','0','0','0'},
            {'0','#','0','0','0','0','0','#','0','#','0','0','#','#','0','#','0','#','#','0'},
            {'0','#','#','0','#','#','#','0','0','0','#','#','0','0','0','#','0','0','#','0'},
            {'0','0','#','0','0','0','0','0','#','0','0','0','0','#','#','0','#','#','#','0'},
            {'#','0','0','#','0','#','#','#','0','#','#','#','#','0','0','0','0','0','0','0'},
            {'#','#','0','0','#','0','0','0','0','0','0','0','#','#','#','#','0','#','#','0'},
            {'#','#','#','0','0','0','#','0','#','0','#','0','0','0','0','0','0','#','0','3'}
        }
    };

    int numMazes = sizeof(mazes) / sizeof(mazes[0]);
    printf("\n\nVersione sequenziale riprogettata:\n");

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Point start, end;
        Node nodes[MAX_NODES];
        
        printMaze(maze);
        int nodeCount = initializeNodes(maze, nodes, &start, &end);

        int path[MAX_NODES];
        int pathLength;

        if (solveMazeIntermediate(nodes, start, end, path, &pathLength)) {
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