#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ROWS 6
#define COLS 7
#define MAX_NODES (ROWS * COLS)
#define NUM_DIRECTIONS 4

typedef struct {
    int row;
    int col;
} Point;

typedef struct {
    Point pos;          // posizione nel labirinto
    bool visited;       // indica se il nodo è stato visitato
    int nodeNum;        // numero assegnato al nodo
    int parentIndex;    // indice del nodo genitore nell'array dei nodi
    bool wall;         // indica se è un muro
} Node;

// Definiamo le direzioni come array di Point
const Point directions[NUM_DIRECTIONS] = {
    {-1, 0},  // SU
    {1, 0},   // GIÙ
    {0, -1},  // SINISTRA
    {0, 1}    // DESTRA
};

bool isValid(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

// Converte coordinate matrice in indice array
int coordToIndex(int row, int col) {
    return row * COLS + col;
}

// Calcola la nuova posizione dato un punto e una direzione
Point getNewPosition(Point current, Point direction) {
    Point newPos = {
        current.row + direction.row,
        current.col + direction.col
    };
    return newPos;
}

// Inizializza l'array dei nodi dal labirinto originale
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

bool solveMaze(Node nodes[], Point start, Point end, int path[], int* pathLength) {
    Node* queue[MAX_NODES];
    int front = 0, rear = 0;
    
    // Aggiungi il nodo di partenza alla coda
    int startIdx = coordToIndex(start.row, start.col);
    nodes[startIdx].visited = true;
    queue[rear++] = &nodes[startIdx];
    
    while (front < rear) {
        Node* current = queue[front++];
        
        // Se abbiamo raggiunto la fine
        if (current->pos.row == end.row && current->pos.col == end.col) {
            // Ricostruisci il percorso
            *pathLength = 0;
            Node* temp = current;
            int startNodeIdx = coordToIndex(start.row, start.col);
            
            // Ricostruisci il percorso fino all'inizio
            while (temp != &nodes[startNodeIdx]) {
                path[(*pathLength)++] = temp->nodeNum;
                temp = &nodes[temp->parentIndex];
            }
            path[(*pathLength)++] = nodes[startNodeIdx].nodeNum;
            
            // Inverti il percorso
            for (int i = 0; i < *pathLength / 2; i++) {
                int temp = path[i];
                path[i] = path[*pathLength - 1 - i];
                path[*pathLength - 1 - i] = temp;
            }
            
            return true;
        }
        
        // Esplora le direzioni possibili usando l'array di Point
        for (int i = 0; i < NUM_DIRECTIONS; i++) {
            Point newPos = getNewPosition(current->pos, directions[i]);
            
            if (isValid(newPos.row, newPos.col)) {
                int newIdx = coordToIndex(newPos.row, newPos.col);
                if (!nodes[newIdx].visited && !nodes[newIdx].wall) {
                    nodes[newIdx].visited = true;
                    nodes[newIdx].parentIndex = coordToIndex(current->pos.row, current->pos.col);
                    queue[rear++] = &nodes[newIdx];
                }
            }
        }
    }
    
    return false;
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
        
        // Stampa il labirinto iniziale
        printMaze(maze);

        // Inizializza e numera i nodi
        int nodeCount = initializeNodes(maze, nodes, &start, &end);

        // Trova e stampa il percorso
        int path[MAX_NODES];
        int pathLength;

        if (solveMaze(nodes, start, end, path, &pathLength)) {
            printf("\nPercorso più breve (sequenza di nodi): ");
            for (int j = 0; j < pathLength; j++) {
                printf("%d", path[j]);
                if (j < pathLength - 1) printf(", ");
            }
            printf("\n");
        } else {
            printf("\nNessun percorso trovato!\n");
        }
        printf("\n.\n");
    }
    return 0;
}