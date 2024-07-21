#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "uthash.h"

#define MAX_PATH 260



// ### NOTE:: NVIDIA USES 'nvcc' compiler- ex: nvcc helloworld.cu -o hello.out




///
typedef struct {
    char fileName[MAX_PATH];  // key
    UT_hash_handle hh;        // makes this structure hashable
} FileEntry;

void ListFilesInDirectory(const char* folderPath);
void PrintFilesToFile(const char* folderPath, const char* outputFilePath);
void SaveFilesToHashmap(const char* folderPath, FileEntry** fileMap);
void CheckDuplicatesInHashmap(FileEntry* fileMap);
void FreeHashmap(FileEntry* fileMap);
void printStringArray(char* arr[], int size)

extern "C" void CheckDuplicatesWithCuda(char** fileNames, int numFiles);

int main() {
    char folderPath[MAX_PATH];
    char outputFilePath[MAX_PATH];

    printf("Enter the folder path to traverse: ");
    fgets(folderPath, MAX_PATH, stdin); // 1. OUTPUT FP. 2. THE UNIQUE KEY. 3 THE PIPE

    // Remove the newline character at the end if it exists
    size_t len = strlen(folderPath);
    if (folderPath[len - 1] == '\n') {
        folderPath[len - 1] = '\0';
    }

    printf("Enter the output file path (e.g., C:\\output.txt): ");
    fgets(outputFilePath, MAX_PATH, stdin); // 1. OUTPUT FP. 2. THE UNIQUE KEY. 3 THE PIPE


    // Remove the newline character at the end if it exists
    len = strlen(outputFilePath);
    if (outputFilePath[len - 1] == '\n') {
        outputFilePath[len - 1] = '\0';
    }


    ListFilesInDirectory(folderPath);
    PrintFilesToFile(folderPath, outputFilePath);

    // Save files to hashmap
    FileEntry* fileMap = NULL;
    SaveFilesToHashmap(folderPath, &fileMap);

    // Collect file names into an array for CUDA processing
    int numFiles = HASH_COUNT(fileMap);
    char** fileNames = (char**)malloc(numFiles * sizeof(char*));
    FileEntry* currentFile;
    int i = 0;

    printf("[!] Collect file names into an array for CUDA processing\n %d numfiles: \n", numFiles);
    /// NOTE: PIPES THE CURRENT FILE INTO DEFINED STRUCT 'fileName'
    for (currentFile = fileMap; currentFile != NULL; currentFile = currentFile->hh.next) {
        fileNames[i] = currentFile->fileName;
        printf(" [!] PROCESSING:  \"%s\"\n", fileNames[i]);
        i++;
    }
    printf("%d [+] numfiles: \n", numFiles);

//    int size = sizeof(fileNames) / sizeof(fileNames[0]);
//    printStringArray(stringArray, size);




    // Check for duplicates using CUDA
    CheckDuplicatesWithCuda(fileNames, numFiles);

    // Free the array
    free(fileNames);

    // Free hashmap
    FreeHashmap(fileMap);

    return 0;
}

void ListFilesInDirectory(const char* folderPath) {
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Create the search pattern
    char searchPattern[MAX_PATH];
    snprintf(searchPattern, MAX_PATH, "%s\\*.*", folderPath);

    // Find the first file in the directory
    hFind = FindFirstFile(searchPattern, &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Error: Unable to open directory %s\n", folderPath);
        return;
    }

    printf("Files in directory %s:\n", folderPath);
    do {
        const char* fileName = findFileData.cFileName;

        // Skip the current and parent directory entries
        if (strcmp(fileName, ".") != 0 && strcmp(fileName, "..") != 0) {
            printf("%s\n", fileName);
        }
    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
}

void PrintFilesToFile(const char* folderPath, const char* outputFilePath) {
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = INVALID_HANDLE_VALUE;
    FILE* outputFile = fopen(outputFilePath, "w");

    if (outputFile == NULL) {
        printf("Error: Unable to open output file %s\n", outputFilePath);
        return;
    }

    // Create the search pattern
    char searchPattern[MAX_PATH];
    snprintf(searchPattern, MAX_PATH, "%s\\*.*", folderPath);

    // Find the first file in the directory
    hFind = FindFirstFile(searchPattern, &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Error: Unable to open directory %s\n", folderPath);
        fclose(outputFile);
        return;
    }

    do {
        const char* fileName = findFileData.cFileName;

        // Skip the current and parent directory entries
        if (strcmp(fileName, ".") != 0 && strcmp(fileName, "..") != 0) {
            fprintf(outputFile, "%s\n", fileName);
        }
    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
    fclose(outputFile);
}

void SaveFilesToHashmap(const char* folderPath, FileEntry** fileMap) {
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Create the search pattern
    char searchPattern[MAX_PATH];
    snprintf(searchPattern, MAX_PATH, "%s\\*.*", folderPath);

    // Find the first file in the directory
    hFind = FindFirstFile(searchPattern, &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        printf("Error: Unable to open directory %s\n", folderPath);
        return;
    }

    do {
        const char* fileName = findFileData.cFileName;

        // Skip the current and parent directory entries
        if (strcmp(fileName, ".") != 0 && strcmp(fileName, "..") != 0) {
            FileEntry* newEntry = (FileEntry*)malloc(sizeof(FileEntry));
            if (newEntry == NULL) {
                printf("Error: Memory allocation failed\n");
                FindClose(hFind);
                return;
            }
            strncpy(newEntry->fileName, fileName, MAX_PATH);
            HASH_ADD_STR(*fileMap, fileName, newEntry);
        }
    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
}

void CheckDuplicatesInHashmap(FileEntry* fileMap) {
    FileEntry* currentFile;
    FileEntry* tmp;
    int duplicateCount = 0;

    // Create a secondary hashmap to check for duplicates
    FileEntry* duplicateMap = NULL;

    HASH_ITER(hh, fileMap, currentFile, tmp) {
        FileEntry* entry;
        HASH_FIND_STR(duplicateMap, currentFile->fileName, entry);
        if (entry) {
            printf("Duplicate found: %s\n", currentFile->fileName);
            duplicateCount++;
        } else {
            // Add to duplicate map
            FileEntry* newEntry = (FileEntry*)malloc(sizeof(FileEntry));
            if (newEntry == NULL) {
                printf("Error: Memory allocation failed\n");
                FreeHashmap(duplicateMap);
                return;
            }
            strncpy(newEntry->fileName, currentFile->fileName, MAX_PATH);
            HASH_ADD_STR(duplicateMap, fileName, newEntry);
        }
    }

    if (duplicateCount == 0) {
        printf("No duplicates found.\n");
    }

    // Free the duplicate map
    FreeHashmap(duplicateMap);
}

void FreeHashmap(FileEntry* fileMap) {
    FileEntry* currentFile;
    FileEntry* tmp;

    HASH_ITER(hh, fileMap, currentFile, tmp) {
        HASH_DEL(fileMap, currentFile);
        free(currentFile);
    }
}


void printStringArray(char* arr[], int size) {
    printf("String Array: [\n");
    for (int i = 0; i < size; i++) {
        // Print each string with quotation marks and on a new line
        printf("  \"%s\"\n", arr[i]);
    }
    printf("]\n");
}