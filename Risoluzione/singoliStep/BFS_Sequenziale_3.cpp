/// BFS_Sequenziale_3.cpp – v1.1 (supporto 100×100)
/// ---------------------------------------------------------------------------
/// • Correzioni per labirinti fino ad almeno 100×100 celle.
///   ‑ Ogni riga del labirinto viene ora allocata con `cols+1` byte per il '\0'.
///   ‑ Copia sicura tramite `strncpy` (no overflow).
/// • Fix variabili riepilogo (numMazes ↔ totMazes).
/// • Timer globale, contatore risolti e lunghezze percorso (come versione CUDA).
/// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <vector>
#include <iostream>

#define NUM_DIRECTIONS 4
#define MAX_MAZES 1000
#define NOT_VISITED -1

// -----------------------------------------------------------------------------
// STRUTTURE DATI
// -----------------------------------------------------------------------------
typedef struct {
    int x, y;
    bool visited;
    int nodeId;
    int parentIndex;
    bool wall, start, goal;
} Node;

typedef struct {
    char** data;
    int rows, cols;
} Maze;

// -----------------------------------------------------------------------------
// UTILITY
// -----------------------------------------------------------------------------
static inline bool isValid(int x,int y,int rows,int cols){return x>=0&&x<rows&&y>=0&&y<cols;}
static inline int  coordToIndex(int x,int y,int cols){return x*cols+y;}

// -----------------------------------------------------------------------------
// ALLOCA / FREE MAZE (ora cols+1)
// -----------------------------------------------------------------------------
Maze* createMaze(int rows,int cols){
    Maze* m=(Maze*)malloc(sizeof(Maze));
    if(!m) return NULL;
    m->rows=rows; m->cols=cols;
    m->data=(char**)malloc(rows*sizeof(char*));
    if(!m->data){ free(m); return NULL; }
    for(int i=0;i<rows;++i){
        m->data[i]=(char*)malloc((cols+1)*sizeof(char)); // +1 per terminatore
        if(!m->data[i]){
            for(int j=0;j<i;++j) free(m->data[j]);
            free(m->data); free(m); return NULL;
        }
    }
    return m;
}
void freeMaze(Maze* m){ if(!m) return; for(int i=0;i<m->rows;++i) free(m->data[i]); free(m->data); free(m); }

// -----------------------------------------------------------------------------
// PRINT FUNCTIONS (unchanged except width 5)
// -----------------------------------------------------------------------------
void printMaze(Maze* maze){ puts("\nLabirinto iniziale:");
    for(int i=0;i<maze->rows;++i){ for(int j=0;j<maze->cols;++j){ putchar(maze->data[i][j]); putchar(' ');} putchar('\n'); } }
void printMazeNumbered(Maze* maze){ puts("\nLabirinto numerato:");
    for(int i=0;i<maze->rows;++i){ for(int j=0;j<maze->cols;++j){ int id=coordToIndex(i,j,maze->cols); if(maze->data[i][j]=='#') printf("### "); else printf("%5d ",id);} putchar('\n'); } }

// -----------------------------------------------------------------------------
// INITIALISE NODES
// -----------------------------------------------------------------------------
int initializeNodes(Maze* maze,Node* nodes){
    int maxNodes=maze->rows*maze->cols;
    for(int i=0;i<maze->rows;++i){ for(int j=0;j<maze->cols;++j){ int idx=coordToIndex(i,j,maze->cols);
        nodes[idx].x=i; nodes[idx].y=j; nodes[idx].nodeId=idx; nodes[idx].visited=false; nodes[idx].parentIndex=NOT_VISITED;
        char ch=maze->data[i][j]; nodes[idx].wall=(ch=='#'); nodes[idx].start=(ch=='2'); nodes[idx].goal=(ch=='3'); }}
    //printMazeNumbered(maze); // opzionale
    return maxNodes;
}

// -----------------------------------------------------------------------------
// BFS SEQUENZIALE (identico a prima)
// -----------------------------------------------------------------------------
void exploreNode(Node* nodes,int nodeIdx,int* nextFrontier,int* nextSize,bool* levelDone,int endIdx,int rows,int cols){
    if(nodeIdx==endIdx){ *levelDone=true; return; }
    static const int dx[4]={-1,1,0,0}; static const int dy[4]={0,0,-1,1};
    int cx=nodes[nodeIdx].x, cy=nodes[nodeIdx].y;
    for(int k=0;k<NUM_DIRECTIONS;++k){ int nx=cx+dx[k], ny=cy+dy[k]; if(isValid(nx,ny,rows,cols)){ int nidx=coordToIndex(nx,ny,cols); if(!nodes[nidx].wall && !nodes[nidx].visited){ nodes[nidx].visited=true; nodes[nidx].parentIndex=nodeIdx; nextFrontier[(*nextSize)++]=nidx; if(nidx==endIdx) *levelDone=true; } } }
}

bool solveMazeIntermediate(Maze* maze,Node* nodes,int* path,int* pathLen){
    int maxNodes=maze->rows*maze->cols;
    int* frontier=(int*)malloc(maxNodes*sizeof(int));
    int* nextFrontier=(int*)malloc(maxNodes*sizeof(int));
    int fSize=0,nSize=0; bool levelDone=false;
    int startIdx=-1,endIdx=-1;
    for(int i=0;i<maxNodes;++i){ if(nodes[i].start) startIdx=i; if(nodes[i].goal) endIdx=i; }
    if(startIdx==-1||endIdx==-1){ puts("Labirinto senza start/goal"); free(frontier); free(nextFrontier); return false; }
    frontier[fSize++]=startIdx; nodes[startIdx].visited=true;
    bool found=false;
    while(fSize>0){ nSize=0; levelDone=false;
        for(int i=0;i<fSize;++i){ exploreNode(nodes,frontier[i],nextFrontier,&nSize,&levelDone,endIdx,maze->rows,maze->cols);} if(levelDone){ found=true; break; }
        int* tmp=frontier; frontier=nextFrontier; nextFrontier=tmp; fSize=nSize; }
    if(found){ *pathLen=0; int cur=endIdx; while(cur!=startIdx){ path[(*pathLen)++]=cur; cur=nodes[cur].parentIndex; } path[(*pathLen)++]=startIdx; // reverse
        for(int i=0;i<*pathLen/2;++i){ int t=path[i]; path[i]=path[*pathLen-1-i]; path[*pathLen-1-i]=t; } }
    free(frontier); free(nextFrontier); return found;
}

// -----------------------------------------------------------------------------
// LOAD MAZES (copie sicure)
// -----------------------------------------------------------------------------
int loadMazesFromFile(const char* filename,Maze*** mazes){
    FILE* f=fopen(filename,"r"); if(!f){ perror("open"); return 0; }
    *mazes=(Maze**)malloc(MAX_MAZES*sizeof(Maze*)); int count=0; char line[1024]; char tmp[1000][1024]; int tmpRows=0;
    while(fgets(line,sizeof(line),f)&&count<MAX_MAZES){ line[strcspn(line,"\r\n")]=0; if(strlen(line)==0){ if(tmpRows>0){ int cols=strlen(tmp[0]); Maze* m=createMaze(tmpRows,cols); if(m){ for(int r=0;r<tmpRows;++r){ strncpy(m->data[r],tmp[r],cols); m->data[r][cols]='\0'; } (*mazes)[count++]=m; } tmpRows=0; } continue; }
        if(tmpRows<1000){ strncpy(tmp[tmpRows],line,1023); tmp[tmpRows][1023]='\0'; ++tmpRows; } }
    if(tmpRows>0){ int cols=strlen(tmp[0]); Maze* m=createMaze(tmpRows,cols); if(m){ for(int r=0;r<tmpRows;++r){ strncpy(m->data[r],tmp[r],cols); m->data[r][cols]='\0'; } (*mazes)[count++]=m; } }
    fclose(f); return count; }

// -----------------------------------------------------------------------------
// PATH LENGTH
// -----------------------------------------------------------------------------
int computePathLength(int* path,int pathLen){ return pathLen>0?pathLen-1:-1; }

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(){
    Maze** mazes; const char* file="mazes1000.txt";
    int numMazes=loadMazesFromFile(file,&mazes); if(numMazes==0){ puts("No mazes loaded."); return 1; }

    std::vector<int> pathLengths(numMazes,-1); int solvedCnt=0;
    auto t0=std::chrono::high_resolution_clock::now();

    for(int i=0;i<numMazes;++i){ Maze* m=mazes[i]; int maxNodes=m->rows*m->cols; Node* nodes=(Node*)malloc(maxNodes*sizeof(Node)); if(!nodes){ puts("alloc nodes fail"); continue; } int* path=(int*)malloc(maxNodes*sizeof(int)); int pathLen=0;
        initializeNodes(m,nodes);
        bool found=solveMazeIntermediate(m,nodes,path,&pathLen);
        if(found){ ++solvedCnt; pathLengths[i]=computePathLength(path,pathLen); }
        free(nodes); free(path); }

    auto t1=std::chrono::high_resolution_clock::now(); 
    double elapsedMs=std::chrono::duration<double,std::milli>(t1-t0).count();

    // riepilogo
    std::cout<<"\n--- RIEPILOGO ---\n"; std::cout<<"Tempo totale: "<<elapsedMs<<" ms\n"; std::cout<<"Labirinti risolti: "<<solvedCnt<<"/"<<numMazes<<"\n"; std::cout<<"Lunghezze percorsi -> ";
    for(int i=0;i<numMazes;++i){ std::cout<<"L"<<i+1<<":"<<pathLengths[i]; if(i+1<numMazes) std::cout<<", "; }
    std::cout<<"\n";

    for(int i=0;i<numMazes;++i) freeMaze(mazes[i]); free(mazes);
    return 0; 
}
