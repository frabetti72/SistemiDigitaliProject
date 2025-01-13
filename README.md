Il progetto sarà così strutturato:
- Cartella di Risoluzione: per algoritmi di risoluzione
- - BFS_Resolution_Sequenziale: per la risoluzione sequenziale
  - BFS_Resolution_CUDA: per la risoluzione con CUDA
  - BFS_Resolution_CUDA_optimized: per la risoluzione ottimizzata

Inoltre sono presenti branch:
- main: branch principale dove vengono pushate solo le modifiche funzionanti
- develop: branch per la sperimentazione
- developR: branch per Frabe
- developF: branch per Filippo

Attualmente presente una cartella (stepsV2) che documenta tutti gli steps di modifica a partire da BFS_Resolution_Sequenziale_V2:
- BFS_Resolution_Sequenziale_Riprogettata: versione riproettata per essere espandibile in parallelo, ma ancora sequenziale
- BFS_CUDA_1: prima versione CUDA, basata su BFS_Resolution_Sequenziale_Riprogettata, si occupa di mettere in parallelo solo l'esplorazione dei nodi adiacenti ad uno scelto assegando un thread a quel nodo (quello da cui deve partire l'esplorazione)

