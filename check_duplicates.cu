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

extern "C" void CheckDuplicatesWithCuda(FileEntry** fileEntries, int