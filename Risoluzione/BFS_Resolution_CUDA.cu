#include <stdio.h>

__global__ void kernel(void) {
   printf("Hello, world! from GPU thread %d\n", threadIdx.x);
}

int main() {
   kernel<<<1, 10>>>(); // Lancia il kernel con 1 blocco e 10 thread

   cudaDeviceSynchronize(); // Aspetta che tutti i thread del kernel finiscano

   return 0;
}