#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#define ROWS 6
#define COLS 7
#define MAX_NODES (ROWS * COLS)
#define BLOCK_SIZE 256
#define MAX_MAZES 10  // Maximum number of mazes to process in parallel

typedef struct {
    int row;
    int col;
} Point;

typedef struct {
    Point pos;
    int nodeNum;
} Node;

// Direction arrays for maze traversal
__constant__ int dr[4] = {-1, 1, 0, 0};
__constant__ int dc[4] = {0, 0, -1, 1};

__device__ bool isValid(int r, int c) {
    return r >= 0 && r < ROWS && c >= 0 && c < COLS;
}

// Kernel to find start and end points in parallel for multiple mazes
__global__ void findStartEndKernel(char* mazes, Point* starts, Point* ends, int numMazes) {
    int mazeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mazeIdx >= numMazes) return;

    char* maze = mazes + mazeIdx * ROWS * COLS;
    
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (maze[i * COLS + j] == '2') {
                starts[mazeIdx].row = i;
                starts[mazeIdx].col = j;
            } else if (maze[i * COLS + j] == '3') {
                ends[mazeIdx].row = i;
                ends[mazeIdx].col = j;
            }
        }
    }
}

// Kernel to number nodes in parallel
__global__ void numberNodesKernel(char* maze, int* nodeMap, int* nodeCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / COLS;
    int col = idx % COLS;
    
    if (row >= ROWS || col >= COLS) return;
    
    nodeMap[idx] = -1;
    if (maze[idx] != '#') {
        nodeMap[idx] = atomicAdd(nodeCount, 1);
    }
}

// Kernel for parallel BFS exploration
__global__ void bfsKernel(char* maze, int* nodeMap, Point start, Point end,
                         bool* visited, Point* parent, Node* queue,
                         int* queueSize, bool* foundPath) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *queueSize) return;
    
    Node current = queue[tid];
    
    if (current.pos.row == end.row && current.pos.col == end.col) {
        *foundPath = true;
        return;
    }
    
    // Each thread explores one direction
    for (int i = 0; i < 4; i++) {
        int newRow = current.pos.row + dr[i];
        int newCol = current.pos.col + dc[i];
        
        if (isValid(newRow, newCol)) {
            int newIdx = newRow * COLS + newCol;
            if (!visited[newIdx] && maze[newIdx] != '#') {
                int queueIdx = atomicAdd(queueSize, 1);
                if (queueIdx < MAX_NODES) {
                    Node newNode;
                    newNode.pos.row = newRow;
                    newNode.pos.col = newCol;
                    newNode.nodeNum = nodeMap[newIdx];
                    queue[queueIdx] = newNode;
                    
                    visited[newIdx] = true;
                    parent[newIdx] = current.pos;
                }
            }
        }
    }
}

// Host function to solve maze using CUDA
bool solveMazeCUDA(char* maze_h, int* nodeMap_h, Point start, Point end,
                   int* path, int* pathLength) {
    char* maze_d;
    int* nodeMap_d;
    bool* visited_d;
    Point* parent_d;
    Node* queue_d;
    int* queueSize_d;
    bool* foundPath_d;
    
    // Allocate device memory
    cudaMalloc(&maze_d, ROWS * COLS * sizeof(char));
    cudaMalloc(&nodeMap_d, ROWS * COLS * sizeof(int));
    cudaMalloc(&visited_d, ROWS * COLS * sizeof(bool));
    cudaMalloc(&parent_d, ROWS * COLS * sizeof(Point));
    cudaMalloc(&queue_d, MAX_NODES * sizeof(Node));
    cudaMalloc(&queueSize_d, sizeof(int));
    cudaMalloc(&foundPath_d, sizeof(bool));
    
    // Copy data to device
    cudaMemcpy(maze_d, maze_h, ROWS * COLS * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(nodeMap_d, nodeMap_h, ROWS * COLS * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize variables
    cudaMemset(visited_d, 0, ROWS * COLS * sizeof(bool));
    cudaMemset(queueSize_d, 0, sizeof(int));
    cudaMemset(foundPath_d, 0, sizeof(bool));
    
    // Add start node to queue - corrected initialization
    Node startNode;
    startNode.pos.row = start.row;
    startNode.pos.col = start.col;
    startNode.nodeNum = nodeMap_h[start.row * COLS + start.col];
    cudaMemcpy(queue_d, &startNode, sizeof(Node), cudaMemcpyHostToDevice);
    
    int initialQueueSize = 1;
    cudaMemcpy(queueSize_d, &initialQueueSize, sizeof(int), cudaMemcpyHostToDevice);
    
    bool foundPath = false;
    int maxIterations = ROWS * COLS;  // Maximum possible iterations
    
    for (int iter = 0; iter < maxIterations && !foundPath; iter++) {
        int currentQueueSize;
        cudaMemcpy(&currentQueueSize, queueSize_d, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (currentQueueSize == 0) break;
        
        int numBlocks = (currentQueueSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bfsKernel<<<numBlocks, BLOCK_SIZE>>>(maze_d, nodeMap_d, start, end,
                                            visited_d, parent_d, queue_d,
                                            queueSize_d, foundPath_d);
        
        cudaMemcpy(&foundPath, foundPath_d, sizeof(bool), cudaMemcpyDeviceToHost);
    }
    
    if (foundPath) {
        // Reconstruct path on host
        *pathLength = 0;
        Point* parent_h = (Point*)malloc(ROWS * COLS * sizeof(Point));
        cudaMemcpy(parent_h, parent_d, ROWS * COLS * sizeof(Point), cudaMemcpyDeviceToHost);
        
        Point pos = end;
        while (pos.row != start.row || pos.col != start.col) {
            path[(*pathLength)++] = nodeMap_h[pos.row * COLS + pos.col];
            pos = parent_h[pos.row * COLS + pos.col];
        }
        path[(*pathLength)++] = nodeMap_h[start.row * COLS + start.col];
        
        // Reverse path
        for (int i = 0; i < *pathLength / 2; i++) {
            int temp = path[i];
            path[i] = path[*pathLength - 1 - i];
            path[*pathLength - 1 - i] = temp;
        }
        
        free(parent_h);
    }
    
    // Cleanup
    cudaFree(maze_d);
    cudaFree(nodeMap_d);
    cudaFree(visited_d);
    cudaFree(parent_d);
    cudaFree(queue_d);
    cudaFree(queueSize_d);
    cudaFree(foundPath_d);
    
    return foundPath;
}

int main() {
    // Array of mazes (same as original code)
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
    
    // Allocate memory for parallel processing of start/end points
    Point* starts_h = (Point*)malloc(numMazes * sizeof(Point));
    Point* ends_h = (Point*)malloc(numMazes * sizeof(Point));
    Point* starts_d, *ends_d;
    char* mazes_d;
    
    cudaMalloc(&starts_d, numMazes * sizeof(Point));
    cudaMalloc(&ends_d, numMazes * sizeof(Point));
    cudaMalloc(&mazes_d, numMazes * ROWS * COLS * sizeof(char));
    
    // Copy mazes to device
    cudaMemcpy(mazes_d, mazes, numMazes * ROWS * COLS * sizeof(char), cudaMemcpyHostToDevice);
    
    // Find start and end points for all mazes in parallel
    int numBlocks = (numMazes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    findStartEndKernel<<<numBlocks, BLOCK_SIZE>>>(mazes_d, starts_d, ends_d, numMazes);
    
    cudaMemcpy(starts_h, starts_d, numMazes * sizeof(Point), cudaMemcpyDeviceToHost);
    cudaMemcpy(ends_h, ends_d, numMazes * sizeof(Point), cudaMemcpyDeviceToHost);

    // Process each maze
    for (int i = 0; i < numMazes; i++) {
        printf("\n\nProcessing maze %d:\n", i + 1);
        
        // Print original maze
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                printf("%c ", mazes[i][r][c]);
            }
            printf("\n");
        }
        
        // Create node map
        int nodeMap[ROWS][COLS];
        int* nodeCount_d;
        cudaMalloc(&nodeCount_d, sizeof(int));
        cudaMemset(nodeCount_d, 0, sizeof(int));
        
        char* currentMaze_d;
        int* nodeMap_d;
        cudaMalloc(&currentMaze_d, ROWS * COLS * sizeof(char));
        cudaMalloc(&nodeMap_d, ROWS * COLS * sizeof(int));
        
        cudaMemcpy(currentMaze_d, mazes[i], ROWS * COLS * sizeof(char), cudaMemcpyHostToDevice);
        
        // Number nodes in parallel
        numBlocks = (ROWS * COLS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        numberNodesKernel<<<numBlocks, BLOCK_SIZE>>>(currentMaze_d, nodeMap_d, nodeCount_d);
        
        cudaMemcpy(nodeMap, nodeMap_d, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Print numbered maze
        printf("\nMaze with numbered nodes:\n");
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (nodeMap[r][c] == -1) {
                    printf("## ");
                } else {
                    printf("%2d ", nodeMap[r][c]);
                }
            }
            printf("\n");
        }
        
        // Find path
        int path[MAX_NODES];
        int pathLength;
        
        if (solveMazeCUDA((char*)mazes[i], (int*)nodeMap, starts_h[i], ends_h[i], path, &pathLength)) {
            printf("\nShortest path (node sequence): ");
            for (int j = 0; j < pathLength; j++) {
                printf("%d", path[j]);
                if (j < pathLength - 1) printf(", ");
            }
            printf("\n");
        } else {
            printf("\nNo path found!\n");
        }
        
        cudaFree(currentMaze_d);
        cudaFree(nodeMap_d);
        cudaFree(nodeCount_d);
    }
    
    // Cleanup
    cudaFree(starts_d);
    cudaFree(ends_d);
    cudaFree(mazes_d);
    free(starts_h);
    free(ends_h);
    
    return 0;
}