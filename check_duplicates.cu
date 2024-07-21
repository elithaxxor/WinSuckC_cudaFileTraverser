#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include "uthash.h"

typedef struct {
    char fileName[260];  // key
    unsigned char* content;   // file content
    size_t contentSize;       // size of content
    UT_hash_handle hh;        // makes this structure hashable
} FileEntry;

__global__ void checkDuplicatesKernel(FileEntry** fileEntries, int numFiles, int* duplicateFlags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numFiles) {
        for (int j = idx + 1; j < numFiles; ++j) {
            if (fileEntries[idx]->contentSize == fileEntries[j]->contentSize &&
                memcmp(fileEntries[idx]->content, fileEntries[j]->content, fileEntries[idx]->contentSize) == 0) {
                duplicateFlags[idx] = 1;
                duplicateFlags[j] = 1;
                }
        }
    }
}

extern "C" void CheckDuplicatesWithCuda(char** fileNames, int numFiles) {

    char** d_fileNames;

    int* d_duplicateFlags;

    int* h_duplicateFlags = (int*)malloc(numFiles * sizeof(int));

    memset(h_duplicateFlags, 0, numFiles * sizeof(int));



    cudaMalloc((void**)&d_fileNames, numFiles * sizeof(char*));

    cudaMalloc((void**)&d_duplicateFlags, numFiles * sizeof(int));

    cudaMemcpy(d_fileNames, fileNames, numFiles * sizeof(char*), cudaMemcpyHostToDevice);

    cudaMemcpy(d_duplicateFlags, h_duplicateFlags, numFiles * sizeof(int), cudaMemcpyHostToDevice);



    int blockSize = 256;

    int numBlocks = (numFiles + blockSize - 1) / blockSize;


    /// TODO: FIX THIS
    checkDuplicatesKernel

}