// Single-source CUDA / HIP portability: the same kernels build with nvcc (for
// local testing on NVIDIA) and hipcc (gfx90a / gfx950). Only the runtime API
// names differ; kernel code is identical.
#pragma once

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define gpuError_t              hipError_t
  #define gpuSuccess              hipSuccess
  #define gpuGetErrorString       hipGetErrorString
  #define gpuMalloc               hipMalloc
  #define gpuFree                 hipFree
  #define gpuMemcpy               hipMemcpy
  #define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
  #define gpuMemcpyToSymbol(sym, src, n)   hipMemcpyToSymbol(HIP_SYMBOL(sym), src, n)
  #define gpuMemcpyFromSymbol(dst, sym, n) hipMemcpyFromSymbol(dst, HIP_SYMBOL(sym), n)
  #define gpuDeviceSynchronize    hipDeviceSynchronize
  #define gpuGetLastError         hipGetLastError
  #define gpuDeviceProp           hipDeviceProp_t
  #define gpuGetDeviceProperties  hipGetDeviceProperties
  // HIP's second launch-bounds argument means waves per SIMD, not blocks per
  // SM, so the baseline leaves AMD occupancy to the compiler: a faithful port,
  // not an AMD-tuned one.
  #define GRIND_LAUNCH_BOUNDS(threads, cuda_min_blocks) __launch_bounds__(threads)
#else
  #include <cuda_runtime.h>
  #define gpuError_t              cudaError_t
  #define gpuSuccess              cudaSuccess
  #define gpuGetErrorString       cudaGetErrorString
  #define gpuMalloc               cudaMalloc
  #define gpuFree                 cudaFree
  #define gpuMemcpy               cudaMemcpy
  #define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
  #define gpuMemcpyToSymbol(sym, src, n)   cudaMemcpyToSymbol(sym, src, n)
  #define gpuMemcpyFromSymbol(dst, sym, n) cudaMemcpyFromSymbol(dst, sym, n)
  #define gpuDeviceSynchronize    cudaDeviceSynchronize
  #define gpuGetLastError         cudaGetLastError
  #define gpuDeviceProp           cudaDeviceProp
  #define gpuGetDeviceProperties  cudaGetDeviceProperties
  #define GRIND_LAUNCH_BOUNDS(threads, cuda_min_blocks) \
      __launch_bounds__(threads, cuda_min_blocks)
#endif

// Defined by the program: under the grind protocol a GPU failure must still be
// reported as an error event, not just an exit code.
[[noreturn]] void gpu_fatal(const char *file, int line, const char *what);

#define GPU_CHECK(call)                                                        \
    do {                                                                       \
        gpuError_t gpu_err_ = (call);                                          \
        if (gpu_err_ != gpuSuccess)                                            \
            gpu_fatal(__FILE__, __LINE__, gpuGetErrorString(gpu_err_));        \
    } while (0)
