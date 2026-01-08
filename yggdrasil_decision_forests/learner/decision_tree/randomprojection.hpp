// If num_cols is large, use global memory instead of shared memory for base_indices.
// Each thread performs its own shuffle.
// You can store the output total_col_indices on the device, then copy back to host as needed.
// If this needs to run repeatedly, set up curandState once per thread for reuse.

#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _status = (call);                                         \
        if (_status != cudaSuccess) {                                         \
            std::cerr << "CUDA ERROR: " << cudaGetErrorString(_status)        \
                      << " (code " << _status << ") "                         \
                      << "in " << __FILE__ << ':' << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>

/* ------------------------------------------------------------------ */
/* Simple error-check wrapper                                          */
#define CUDA_CHECK(x)  do {                                           \
    cudaError_t _err = (x);                                           \
    if (_err != cudaSuccess) {                                        \
        fprintf(stderr,"CUDA error %s (%d) at %s:%d\n",               \
                cudaGetErrorString(_err), _err, __FILE__, __LINE__);  \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

/* ------------------------------------------------------------------ */
/* Three global work streams                                           */
/* inline  ⇒ one definition across all translation units (C++17)       */
#if CUDART_VERSION >= 11020
  inline cudaStream_t gStreamA = nullptr;
  inline cudaStream_t gStreamB = nullptr;
  inline cudaStream_t gStreamC = nullptr;
#else
  /* Toolkit < 11.2 – streams unused but keep the symbols */
  inline cudaStream_t gStreamA = nullptr;
  inline cudaStream_t gStreamB = nullptr;
  inline cudaStream_t gStreamC = nullptr;
#endif

/* ------------------------------------------------------------------ */
/* create / destroy helpers                                            */
inline void createWorkStreams()
{
#if CUDART_VERSION >= 11020
    static std::once_flag once;
    std::call_once(once, []{
        CUDA_CHECK( cudaStreamCreateWithFlags(&gStreamA,
                                              cudaStreamNonBlocking) );
        CUDA_CHECK( cudaStreamCreateWithFlags(&gStreamB,
                                              cudaStreamNonBlocking) );
        CUDA_CHECK( cudaStreamCreateWithFlags(&gStreamC,
                                              cudaStreamNonBlocking) );
    });
#endif
}

inline void destroyWorkStreams()
{
#if CUDART_VERSION >= 11020
    if (gStreamA) { CUDA_CHECK(cudaStreamDestroy(gStreamA)); gStreamA=nullptr; }
    if (gStreamB) { CUDA_CHECK(cudaStreamDestroy(gStreamB)); gStreamB=nullptr; }
    if (gStreamC) { CUDA_CHECK(cudaStreamDestroy(gStreamC)); gStreamC=nullptr; }
#endif
}

/* ------------------------------------------------------------------ */
/* Helper macros – async if CUDA ≥ 11.2, legacy otherwise              */
#if CUDART_VERSION >= 11020    /* async allocator available ---------- */

  #define DEV_MALLOC_A(ptr,bytes)  do { createWorkStreams();           \
      CUDA_CHECK(cudaMallocAsync((void**)(ptr), (bytes), gStreamA)); } while (0)
  #define DEV_FREE_A(ptr)          CUDA_CHECK(cudaFreeAsync((ptr), gStreamA))
  #define MEMSET_A(ptr,val,bytes)  CUDA_CHECK(cudaMemsetAsync((ptr), (val), (bytes), gStreamA))
  #define MEMCPY_A(dst,src,bytes,kind) CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), (kind), gStreamA))

  #define DEV_MALLOC_B(ptr,bytes)  do { createWorkStreams();           \
      CUDA_CHECK(cudaMallocAsync((void**)(ptr), (bytes), gStreamB)); } while (0)
  #define DEV_FREE_B(ptr)          CUDA_CHECK(cudaFreeAsync((ptr), gStreamB))
  #define MEMSET_B(ptr,val,bytes)  CUDA_CHECK(cudaMemsetAsync((ptr), (val), (bytes), gStreamB))
  #define MEMCPY_B(dst,src,bytes,kind) CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), (kind), gStreamB))

  #define DEV_MALLOC_C(ptr,bytes)  do { createWorkStreams();           \
      CUDA_CHECK(cudaMallocAsync((void**)(ptr), (bytes), gStreamC)); } while (0)
  #define DEV_FREE_C(ptr)          CUDA_CHECK(cudaFreeAsync((ptr), gStreamC))
  #define MEMSET_C(ptr,val,bytes)  CUDA_CHECK(cudaMemsetAsync((ptr), (val), (bytes), gStreamC))
  #define MEMCPY_C(dst,src,bytes,kind) CUDA_CHECK(cudaMemcpyAsync((dst), (src), (bytes), (kind), gStreamC))

#else                           /* toolkit < 11.2 : legacy fallback --- */
  #define DEV_MALLOC_A(ptr,bytes) CUDA_CHECK(cudaMalloc((void**)(ptr), (bytes)))
  #define DEV_FREE_A(ptr)         CUDA_CHECK(cudaFree((ptr)))
  #define MEMSET_A(ptr,val,bytes) CUDA_CHECK(cudaMemset((ptr), (val), (bytes)))
  #define MEMCPY_A(dst,src,bytes,kind) CUDA_CHECK(cudaMemcpy((dst), (src), (bytes), (kind)))

  #define DEV_MALLOC_B(ptr,bytes) DEV_MALLOC_A(ptr,bytes)
  #define DEV_FREE_B(ptr)         DEV_FREE_A(ptr)
  #define MEMSET_B(ptr,val,bytes) MEMSET_A(ptr,val,bytes)
  #define MEMCPY_B(dst,src,bytes,kind) MEMCPY_A(dst,src,bytes,kind)

  #define DEV_MALLOC_C(ptr,bytes) DEV_MALLOC_A(ptr,bytes)
  #define DEV_FREE_C(ptr)         DEV_FREE_A(ptr)
  #define MEMSET_C(ptr,val,bytes) MEMSET_A(ptr,val,bytes)
  #define MEMCPY_C(dst,src,bytes,kind) MEMCPY_A(dst,src,bytes,kind)
#endif

void warmupfunction();

void ApplyProjectionColumnADD (const float* d_flat_data,
                                const unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<int>>& projection_col_idx,
                                std::vector<std::vector<float>>& projection_weights,
                                const int num_rows,  //num_rows
                                const int num_proj, //num_proj
                                const int train_dataset,
                                double* elapsed_ms,
                                const int split_method,
                                const bool verbose
                              );

void EqualWidthHistogram (const float* __restrict__ d_col_add_projected, //attributes
const unsigned int* __restrict__ selected_examples, //selected examples
const unsigned int* __restrict__ d_global_labels_data,
float* d_min_vals,
float* d_max_vals,
float* h_min_vals,
float* h_max_vals,
float* d_bin_widths,
int** h_hist0,
int** h_hist1,
int** h_hist2,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj
);

void RandomHistogram (const float* __restrict__ d_col_add_projected, //attributes
const unsigned int* __restrict__ selected_examples, //selected examples
const unsigned int* __restrict__ d_global_labels_data,
float* h_min_vals,
float* h_max_vals,
int** h_hist0,
int** h_hist1,
int** h_hist2,
float** d_candidate_splits,
const int num_rows, //selected_examples.size()
const int num_bins,
const int num_proj,
std::mt19937& random
);
                    
void HistogramSplit (const int* d_hist_class0,
                    const int* d_hist_class1,
                    const int* d_hist_class2,
                    const float* d_candidate_splits,
                    const float* h_min_vals,
                    const float* d_bin_widths,
                    const int num_proj,
                    const int num_bins,
                    const int num_rows,
                    int* best_proj,
                    int* best_bin_out,
                    float* best_gain_out,
                    float* best_threshold_out,
                    int* num_pos_examples_out,
                    double* elapsed_ms,
                    bool verbose,
                    const int comp_method,
                    const int split_method
                    );

void ThrustSortIndicesOnly(float* d_proj_values, 
                          unsigned int* d_row_ids,
                          unsigned int* d_selected_examples, 
                          int num_rows, 
                          int num_proj);

void ExactSplit(
    unsigned int* d_sorted_indices,  // [num_proj * num_rows]
    const unsigned int* d_labels,          // [num_proj * num_rows]
    float* best_gain_out, // [num_proj], initial best gain values
    int* best_split_out, // [num_proj], initial best split values
    float* best_threshold_out,
    int* best_proj,
    const int num_rows,
    const int num_proj,
    float* d_col_add_projected,  // [num_proj * num_rows]
    double* elapsed_ms,
    bool verbose,
    const int comp_method
    );

