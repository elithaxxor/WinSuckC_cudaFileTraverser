cmake_minimum_required(VERSION 3.28)
project(WinSuckC_cudaFileTraverser CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(WinSuckC_cudaFileTraverser STATIC library.cu)

set_target_properties(WinSuckC_cudaFileTraverser PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
