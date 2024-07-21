#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "uthash.h"

typedef struct {
    char fileName[260];  // key
    unsigned char* content;   // file content
    size_t contentSize;       // size of content
    UT_hash_handle hh;        // makes this structure hashable
} HashEntry;

__global__ void checkDuplicatesBinaryKernel(HashEntry** fileEntries, int numBinary, int* duplicateFlags) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBinary) {
        for (int j = idx + 1; j < numBinary; ++j) {
            if (fileEntries[idx]->contentSize == fileEntries[j]->contentSize &&
                memcmp(fileEntries[idx]->content, fileEntries[j]->content, fileEntries[idx]->contentSize) == 0) {
                duplicateFlags[idx] = 1;
                duplicateFlags[j] = 1;
                }
        }
    }
}

extern "C" void CheckDuplicateBinaryWithCuda(HashEntry** fileEntries, int numBinary) {

    char** d_fileNames;
    int* d_duplicateFlags;
    int* h_duplicateFlags = (int*)malloc(numBinary * sizeof(int));
    memset(h_duplicateFlags, 0, numBinary * sizeof(int));


    cudaMalloc((void**)&d_fileNames, numBinary * sizeof(char*));
    cudaMalloc((void**)&d_duplicateFlags, numBinary * sizeof(int));
    cudaMemcpy(d_fileNames, fileNames, numBinary * sizeof(char*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_duplicateFlags, h_duplicateFlags, numBinary * sizeof(int), cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (numBinary + blockSize - 1) / blockSize;

    /// TODO: FIX THIS
    checkDuplicatesKernel

}