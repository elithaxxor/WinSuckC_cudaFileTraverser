# CUDA Duplicate File Checker

This C program uses CUDA to check for duplicate files based on their binary content. It utilizes GPU parallel processing to speed up the comparison of large numbers of files. The program leverages the `uthash` library for hash table management and CUDA for parallel processing.

## Features

- **Parallel Processing**: Uses CUDA to compare files in parallel, significantly speeding up the process.
- **Hash Table Management**: Utilizes `uthash` for efficient storage and lookup of file entries.
- **Binary Content Comparison**: Compares files based on their binary content to identify duplicates.

## Requirements

- CUDA Toolkit installed on your system.
- A CUDA-compatible GPU.
- `uthash` library included in the project.

## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/cuda-duplicate-file-checker.git
   cd cuda-duplicate-file-checker

## Build Notes (functions) 
  FileEntry Structure --> Defines the structure to hold file information for the hash table.

  checkDuplicatesKernel --> CUDA kernel function to check for duplicate files in parallel.

  CheckDuplicatesWithCuda --> Host function to manage memory allocation, data transfer, and kernel launch for duplicate file checking.

## Libraries Used

	•	cuda_runtime.h: For CUDA runtime functions.
	•	uthash.h: For hash table management.
