#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ROWS 6
#define COLS 7
#define MAX_NODES (ROWS * COLS)

typedef struct {
    int row;
    int col;
} Point;

typedef struct {
    Point pos;
    int nodeNum;
} Node;

// Direzioni possibili: su, giù, sinistra, destra
int dr[] = {-1, 1, 0, 0};
int dc[] = {0, 0, -1, 1};

bool isValid(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

// Trova i punti di partenza e arrivo nel labirinto
void findStartEnd(char maze[ROWS][COLS], Point* start, Point* end) {
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (maze[i][j] == '2') {
                start->row = i;
                start->col = j;
            } else if (maze[i][j] == '3') {
                end->row = i;
                end->col = j;
            }
        }
    }
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

int numberNodes(char maze[ROWS][COLS], int nodeMap[ROWS][COLS]) {
    int nodeCount = 0;
    
    // Inizializza nodeMap a -1
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            nodeMap[i][j] = -1;
        }
    }
    
    // Assegna numeri progressivi alle celle percorribili
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (maze[i][j] != '#') {
                nodeMap[i][j] = nodeCount++;
            }
        }
    }
    
    // Stampa il labirinto con i numeri dei nodi
    printf("\nLabirinto con nodi numerati:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (nodeMap[i][j] == -1) {
                printf("## ");
            } else {
                printf("%2d ", nodeMap[i][j]);
            }
        }
        printf("\n");
    }
    
    return nodeCount;
}

bool solveMaze(char maze[ROWS][COLS], int nodeMap[ROWS][COLS], 
               Point start, Point end, int path[], int* pathLength) {
    bool visited[ROWS][COLS] = {false};
    Point parent[ROWS][COLS];
    Node queue[MAX_NODES];
    int front = 0, rear = 0;
    
    // Aggiungi il nodo di partenza alla coda
    queue[rear++] = (Node){{start.row, start.col}, nodeMap[start.row][start.col]};
    visited[start.row][start.col] = true;
    
    while (front < rear) {
        Node current = queue[front++];
        
        // Se abbiamo raggiunto la fine
        if (current.pos.row == end.row && current.pos.col == end.col) {
            // Ricostruisci il percorso
            *pathLength = 0;
            Point pos = current.pos;
            while (pos.row != start.row || pos.col != start.col) {
                path[(*pathLength)++] = nodeMap[pos.row][pos.col];
                pos = parent[pos.row][pos.col];
            }
            path[(*pathLength)++] = nodeMap[start.row][start.col];
            
            // Inverti il percorso (da inizio a fine)
            for (int i = 0; i < *pathLength / 2; i++) {
                int temp = path[i];
                path[i] = path[*pathLength - 1 - i];
                path[*pathLength - 1 - i] = temp;
            }
            
            return true;
        }
        
        // Esplora le direzioni possibili
        for (int i = 0; i < 4; i++) {
            int newRow = current.pos.row + dr[i];
            int newCol = current.pos.col + dc[i];
            
            if (isValid(newRow, newCol) && 
                !visited[newRow][newCol] && 
                maze[newRow][newCol] != '#') {
                Point newPos = {newRow, newCol};
                queue[rear++] = (Node){newPos, nodeMap[newRow][newCol]};
                visited[newRow][newCol] = true;
                parent[newRow][newCol] = current.pos;
            }
        }
    }
    
    return false;
}

int main() {
    // Array di labirinti
    char mazes[][ROWS][COLS] = {
        {
            {'0','0','#','0','0','0','0'},
            {'0','2','#','0','0','0','0'},
            {'0','0','#','0','0','0','0'},
            {'0','0','0','#','0','3','0'},
            {'0','#','#','#','#','0','0'},
            {'0','0','0','0','0','0','0'}
        },//labirinto con percorso lungo
        {
            {'0','0','0','0','0','0','0'},
            {'0','2','#','#','#','#','0'},
            {'0','0','0','0','0','#','0'},
            {'0','#','#','#','0','#','0'},
            {'0','0','0','0','0','3','0'},
            {'0','#','#','#','#','0','0'}
        },//labirinto con percorso corto
        {
            {'0','0','0','0','0','#','0'},
            {'0','2','#','#','#','#','0'},
            {'0','0','0','0','0','#','0'},
            {'0','#','#','#','0','#','0'},
            {'0','0','0','0','#','3','0'},
            {'0','#','#','#','#','0','0'}
        }//labirinto senza percorso        
        
    };

    int numMazes = sizeof(mazes) / sizeof(mazes[0]);

    for (int i = 0; i < numMazes; i++) {
        printf("\n\nTesting maze %d:\n", i + 1);
        char (*maze)[COLS] = mazes[i];

        Point start, end;
        findStartEnd(maze, &start, &end);

        // Stampa il labirinto iniziale
        printMaze(maze);

        // Crea e stampa la mappa dei nodi
        int nodeMap[ROWS][COLS];
        numberNodes(maze, nodeMap);

        // Trova e stampa il percorso
        int path[MAX_NODES];
        int pathLength;

        if (solveMaze(maze, nodeMap, start, end, path, &pathLength)) {
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
