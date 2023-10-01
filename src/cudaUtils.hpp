#pragma once
#ifndef CUDAUTILS_H
#define CUDAUTILS_H
#include <vector>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line);

template<typename T>
void cudaSafeFree(T*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

#endif // !CUDAUTILS_H