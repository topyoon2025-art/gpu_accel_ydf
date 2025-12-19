//randomprojection.cuh
// Online C++ compiler to run C++ program online
#include <cuda.h>
#include <cstddef>  // for std::size_t

#include <cstdlib>   // for rand()
#include <ctime>     // for time()
#include <random>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstring>
// #include "randomprojection.hpp"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <vector_types.h>
// #include "absl/container/btree_set.h"
// #include "absl/log/log.h"
// #include "absl/status/status.h"
// #include "absl/status/statusor.h"
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/sequence.h>


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


// Forward declarations
__global__ void FindBestGiniSplitVariableWidthKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* bin_boundaries,
    const int num_proj,
    const int num_bins,
    float* gini_out_per_bin_per_proj,
    float* threshold_out_per_bin_per_proj
);

void EqualWidthSplit(
    // const int* d_prefix_0, // reserved by YDF - unnecessary
    const int* d_prefix_1,
    const int* d_prefix_2,
    const float* d_min_vals,
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
    const bool verbose,
    const int comp_method
);

void BuildHistogramAndFindBestSplit(
    const float* d_col_add_projected,
    const unsigned int* d_selected_examples,
    const unsigned int* d_global_labels_data,
    float* d_min_vals,
    float* d_max_vals,
    float* d_bin_widths,
    const int num_rows,
    const int num_bins,
    const int num_proj,
    int* best_proj,
    int* best_bin_out,
    float* best_gain_out,
    float* best_threshold_out,
    int* num_pos_examples_out,
    const bool use_variable_width,
    const int comp_method
);

__global__ void warmup() {
    // Empty kernel for warm-up
}

void warmupfunction() {
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();
}


__global__ void ColumnAddProjectionKernel(
    const float* __restrict__ dataset,
    const unsigned int* d_selected_examples,                 
    float* projected, 
    const int* __restrict__ col_offset, //number of columns per projection  
    const int* __restrict__ flat_col_data, //flattened column indices for all projections
    const float* __restrict__ flat_weights, //flattened weights for all projections              
    const int num_selected_examples,
    const int num_total_rows,
    const int num_proj) 
{
    int TPB = blockDim.x;  // threads per block
    
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= num_proj || row >= num_selected_examples) return;

    int selected_features_count = col_offset[col + 1] - col_offset[col];
    int offset = col_offset[col];

    float sum = 0.0f;
    unsigned int example_idx = d_selected_examples[row];

    for (int i = 0; i < selected_features_count; ++i) {
        int feature_idx = flat_col_data[offset + i];
        float weight = flat_weights[offset + i];
        sum += weight * dataset[feature_idx * num_total_rows + example_idx];
    }

    projected[col * num_selected_examples + row] = sum;
}

__global__ void ComputeMinMaxKernel(
    const float* __restrict__ d_col_add_projected,
    float* __restrict__ d_block_min,
    float* __restrict__ d_block_max,
    const int num_rows,
    const int num_proj
){
    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;

    int proj_id = blockIdx.y;
    int tid     = threadIdx.x;
    if (proj_id >= num_proj) return;

    int base_idx = proj_id * num_rows;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // strided loop over rows
    for (int i = blockIdx.x * blockDim.x + tid; i < num_rows; i += gridDim.x * blockDim.x) {
        float val = d_col_add_projected[base_idx + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // store per-thread local min/max in shared
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) {
        int idx = proj_id * gridDim.x + blockIdx.x;
        d_block_min[idx] = shared_min[0];
        d_block_max[idx] = shared_max[0];
    }
}
__global__ void ColumnAddComputeMinMaxCombined(
    const float* __restrict__ dataset,
    const unsigned int* d_selected_examples,                 
    float* projected, 
    const int* __restrict__ col_offset, //number of columns per projection  
    const int* __restrict__ flat_col_data, //flattened column indices for all projections
    const float* __restrict__ flat_weights, //flattened weights for all projections              
    const int num_selected_examples,
    const int num_total_rows,
    const int num_proj,
    float* __restrict__ d_block_min,
    float* __restrict__ d_block_max
    ) 
{
    int TPB = blockDim.x;  // threads per block
    
    int col = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];
    float* shared_min = shared;
    float* shared_max = shared + blockDim.x;

    if (col >= num_proj || row >= num_selected_examples) return;

    int selected_features_count = col_offset[col + 1] - col_offset[col];
    int offset = col_offset[col];

    float sum = 0.0f;
    unsigned int example_idx = d_selected_examples[row];

    for (int i = 0; i < selected_features_count; ++i) {
        int feature_idx = flat_col_data[offset + i];
        float weight = flat_weights[offset + i];
        sum += weight * dataset[feature_idx * num_total_rows + example_idx];
    }

    projected[col * num_selected_examples + row] = sum;
    __syncthreads();

   
    int tid     = threadIdx.x;
    if (col >= num_proj) return;

    int base_idx = col * num_selected_examples;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // strided loop over rows
    for (int i = blockIdx.x * blockDim.x + tid; i < num_selected_examples; i += gridDim.x * blockDim.x) {
        float val = projected[base_idx + i];
        local_min = fminf(local_min, val);
        local_max = fmaxf(local_max, val);
    }

    // store per-thread local min/max in shared
    shared_min[tid] = local_min;
    shared_max[tid] = local_max;
    __syncthreads();

    // block-wide reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    // write block result
    if (tid == 0) {
        int idx = col * gridDim.x + blockIdx.x;
        d_block_min[idx] = shared_min[0];
        d_block_max[idx] = shared_max[0];
    }
}

void ApplyProjectionColumnADD (const float* d_flat_data,
                                const unsigned int* d_selected_examples,//selected examples indices
                                float* d_col_add_projected,
                                float** d_min_vals_out,
                                float** d_max_vals_out,
                                float** d_bin_widths_out,
                                std::vector<std::vector<int>>& projection_col_idx,
                                std::vector<std::vector<float>>& projection_weights,
                                const int num_selected_examples,  //num_rows
                                const int num_proj, //num_proj
                                const int num_total_rows,
                                double* elapsed_apply_ms,
                                const int split_method,
                                const bool verbose
                              )
{

    CUDA_CHECK(cudaGetLastError()); 
    ////////////////////////Data Preparation for col per projection on Host///////////////////////////
    auto startDataPrep = std::chrono::high_resolution_clock::now();
    int result_size = num_selected_examples * num_proj;
    const int P = static_cast<int>(projection_col_idx.size());
    std::vector<int> col_per_proj(P); //Number of columns per projection
    //auto startCPY = std::chrono::high_resolution_clock::now();
    std::transform(projection_col_idx.begin(), projection_col_idx.end(),
                col_per_proj.begin(),
                [](const auto& v) { return static_cast<int>(v.size()); });
    // auto endCPY = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> cpy_duration = endCPY - startCPY;
    // printf("Data Processing Col Per Proj Time taken: %f ms\n", cpy_duration.count());
    ////////////////////////////////////////////////////////////////////////

    //////////////////////exclusive scan to get offsets///////////////////////////
    std::vector<int> offset(col_per_proj.size() + 1);
    /*  offset[0] must be 0.  Write the exclusive scan starting at offset[1]. */
    offset[0] = 0;
    //auto startScan = std::chrono::high_resolution_clock::now();
    if (P > 0) {
        std::inclusive_scan(col_per_proj.begin(),           // first
                        col_per_proj.end(),             // last (exclusive)
                        offset.begin() + 1);    
    }
    // auto endScan = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> scan_duration = endScan - startScan;
    // printf("Exclusive Scan Time taken: %f ms\n", scan_duration.count());  
    ////////////////////////////////////////////////////////////////////////  

    //////////////////////calculate total size for flattening///////////////////////////
    size_t total_size = 0;
    //auto startAcc = std::chrono::high_resolution_clock::now();
    total_size = std::accumulate(
    projection_col_idx.begin(), projection_col_idx.end(), std::size_t{0},
    [](std::size_t sum, const auto& v) { return sum + v.size(); }); // Accumulate total size
    // auto endAcc = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> acc_duration = endAcc - startAcc;
    // printf("Total Size Accumulate Time taken: %f ms\n", acc_duration.count());
    //////////////////////////////////////////////////////////////////////////////////////

    ///////////////////////copy and flatten projection data structures///////////////////////////
    std::vector<int> flat_projection_col_idx(total_size);
    //auto startMemcpy = std::chrono::high_resolution_clock::now();
    int* dst = flat_projection_col_idx.data();
    for (const auto& v : projection_col_idx) {
        const std::size_t bytes = v.size() * sizeof(int); //Flattening column indices
        std::memcpy(dst, v.data(), bytes);   // one bulk copy per inner vector
        dst += v.size();
    }
    // auto endMemcpy = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> memcpy_duration = endMemcpy - startMemcpy;
    // printf("Memcpy Flatten Time taken: %f ms\n", memcpy_duration.count());

    std::vector<float> flat_projection_weights(total_size);
    //auto startMemcpyW = std::chrono::high_resolution_clock::now();
    float* dst_w = flat_projection_weights.data();
    for (const auto& v : projection_weights) {
        std::memcpy(dst_w, v.data(), v.size() * sizeof(float)); //Flattening weights
        dst_w += v.size();
    }
     
    // auto endMemcpyW = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> memcpyW_duration = endMemcpyW - startMemcpyW;
    // printf("Memcpy Flatten Weights Time taken: %f ms\n", memcpyW_duration.count());
    ///////////////////////////////////////////////////////////////////////////////////////////

    // auto endDataPrep = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> data_prep_duration = endDataPrep - startDataPrep;
    //printf("Total Data Preparation Time taken in ApplyProjection: %f ms\n", data_prep_duration.count());
    /////////////////////////////////////////////////End of Data Preparation//////////////////////////////////////

    
    ///////////////////////GPU Memory Allocation and Copy///////////////////////////  

    int* d_offset = nullptr; 
    int* d_flat_projection_col_idx = nullptr;
    float* d_flat_projection_weights = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_offset, offset.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_flat_projection_col_idx, flat_projection_col_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_flat_projection_weights, flat_projection_weights.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_offset, offset.data(), offset.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_projection_col_idx, flat_projection_col_idx.data(), flat_projection_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice));    
    CUDA_CHECK(cudaMemcpy(d_flat_projection_weights, flat_projection_weights.data(), flat_projection_weights.size() * sizeof(float), cudaMemcpyHostToDevice));  

    // Launch CUDA kernel
    dim3 blockDim(256);
    dim3 gridDim((num_selected_examples + blockDim.x - 1) / blockDim.x, num_proj);
    
    if (split_method == 0) {

    ColumnAddProjectionKernel<<<gridDim, blockDim>>>(d_flat_data,
                                                    d_selected_examples,
                                                    d_col_add_projected,
                                                    d_offset,
                                                    d_flat_projection_col_idx,
                                                    d_flat_projection_weights,
                                                    num_selected_examples,
                                                    num_total_rows,
                                                    num_proj);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    }
    else if (split_method == 2) {
        const int total_blocks = num_proj * gridDim.x;
        float* d_min_vals;
        float* d_max_vals;          
        float* d_bin_widths;
        CUDA_CHECK(cudaMalloc(&d_min_vals, num_proj * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_max_vals, num_proj * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bin_widths, num_proj * sizeof(float)));
        *d_min_vals_out = d_min_vals;
        *d_max_vals_out = d_max_vals;
        *d_bin_widths_out = d_bin_widths;

        float* d_block_min;
        float* d_block_max;
        CUDA_CHECK(cudaMalloc(&d_block_min, total_blocks * sizeof(float)));  // Allocate intermediate buffers
        CUDA_CHECK(cudaMalloc(&d_block_max, total_blocks * sizeof(float)));  // Allocate intermediate buffers

        size_t shmem = 2 * blockDim.x * sizeof(float);
    //  ColumnAddProjectionKernel<<<gridDim, blockDim>>>(d_flat_data,
    //                                                 d_selected_examples,
    //                                                 d_col_add_projected,
    //                                                 d_offset,
    //                                                 d_flat_projection_col_idx,
    //                                                 d_flat_projection_weights,
    //                                                 num_selected_examples,
    //                                                 num_total_rows,
    //                                                 num_proj);
    //  CUDA_CHECK(cudaPeekAtLastError());
    //  CUDA_CHECK(cudaDeviceSynchronize());
        // Launch Pass 1: Find min/max per block per projection
        // ComputeMinMaxKernel<<<gridDim, blockDim, shmem>>>(d_col_add_projected,
        //                                             d_block_min,
        //                                             d_block_max,
        //                                             num_selected_examples,
        //                                             num_proj);
        // CUDA_CHECK(cudaPeekAtLastError());
        // CUDA_CHECK(cudaDeviceSynchronize());
        ColumnAddComputeMinMaxCombined<<<gridDim, blockDim, shmem>>>(d_flat_data,
                                                                        d_selected_examples,
                                                                        d_col_add_projected,
                                                                        d_offset,
                                                                        d_flat_projection_col_idx,
                                                                        d_flat_projection_weights,
                                                                        num_selected_examples,
                                                                        num_total_rows,
                                                                        num_proj,
                                                                        d_block_min,
                                                                        d_block_max);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto startCub = std::chrono::high_resolution_clock::now();
        cudaStream_t stream1 = 0;;
        thrust::device_vector<int> d_begin(num_proj);
        thrust::device_vector<int> d_end(num_proj);

        // Fill begin offsets: 0, num_blocks, 2*num_blocks, ...
        thrust::sequence(d_begin.begin(), d_begin.end(), 0, (int)gridDim.x);

        // Fill end offsets: num_blocks, 2*num_blocks, 3*num_blocks, ...
        thrust::sequence(d_end.begin(), d_end.end(), (int)gridDim.x, (int)gridDim.x);

        int* d_begin_ptr = thrust::raw_pointer_cast(d_begin.data());
        int* d_end_ptr   = thrust::raw_pointer_cast(d_end.data());

        // ====================================================================
        // 🚀 CUB: MIN per projection
        // ====================================================================
        void* d_temp = nullptr;
        size_t temp_bytes = 0;

        cub::DeviceSegmentedReduce::Min(
            d_temp, temp_bytes,
            d_block_min,
            d_min_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            stream1
        );
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSegmentedReduce::Min(
            d_temp, temp_bytes,
            d_block_min,
            d_min_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            stream1
        );
        cudaFree(d_temp);

        // ====================================================================
        // 🚀 CUB: MAX per projection
        // ====================================================================
        d_temp = nullptr;
        temp_bytes = 0;
        cub::DeviceSegmentedReduce::Max(
            d_temp, temp_bytes,
            d_block_max,
            d_max_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            stream1
        );
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSegmentedReduce::Max(
            d_temp, temp_bytes,
            d_block_max,
            d_max_vals,
            num_proj,
            d_begin_ptr,
            d_end_ptr,
            stream1
        );
        cudaFree(d_temp);
        
        CUDA_CHECK(cudaFree(d_block_min));
        CUDA_CHECK(cudaFree(d_block_max));
    }
else if (split_method == 3) {  // Random Binary Search
    const int total_blocks = num_proj * gridDim.x;
    float* d_min_vals;
    float* d_max_vals;
    CUDA_CHECK(cudaMalloc(&d_min_vals, num_proj * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_vals, num_proj * sizeof(float)));
    *d_min_vals_out = d_min_vals;
    *d_max_vals_out = d_max_vals;
    *d_bin_widths_out = nullptr;  // Not used for variable-width

    float* d_block_min;
    float* d_block_max;
    CUDA_CHECK(cudaMalloc(&d_block_min, total_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_max, total_blocks * sizeof(float)));

    size_t shmem = 2 * blockDim.x * sizeof(float);
    
    // Combined kernel that does projection and min/max computation
    ColumnAddComputeMinMaxCombined<<<gridDim, blockDim, shmem>>>(
        d_flat_data,
        d_selected_examples,
        d_col_add_projected,
        d_offset,
        d_flat_projection_col_idx,
        d_flat_projection_weights,
        num_selected_examples,
        num_total_rows,
        num_proj,
        d_block_min,
        d_block_max);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Use CUB to find final min/max per projection
    cudaStream_t stream1 = 0;
    thrust::device_vector<int> d_begin(num_proj);
    thrust::device_vector<int> d_end(num_proj);

    // Fill begin offsets: 0, num_blocks, 2*num_blocks, ...
    thrust::sequence(d_begin.begin(), d_begin.end(), 0, (int)gridDim.x);

    // Fill end offsets: num_blocks, 2*num_blocks, 3*num_blocks, ...
    thrust::sequence(d_end.begin(), d_end.end(), (int)gridDim.x, (int)gridDim.x);

    int* d_begin_ptr = thrust::raw_pointer_cast(d_begin.data());
    int* d_end_ptr   = thrust::raw_pointer_cast(d_end.data());

    // ====================================================================
    // CUB: MIN per projection
    // ====================================================================
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    cub::DeviceSegmentedReduce::Min(
        d_temp, temp_bytes,
        d_block_min,
        d_min_vals,
        num_proj,
        d_begin_ptr,
        d_end_ptr,
        stream1
    );
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedReduce::Min(
        d_temp, temp_bytes,
        d_block_min,
        d_min_vals,
        num_proj,
        d_begin_ptr,
        d_end_ptr,
        stream1
    );
    cudaFree(d_temp);

    // ====================================================================
    // CUB: MAX per projection
    // ====================================================================
    d_temp = nullptr;
    temp_bytes = 0;
    cub::DeviceSegmentedReduce::Max(
        d_temp, temp_bytes,
        d_block_max,
        d_max_vals,
        num_proj,
        d_begin_ptr,
        d_end_ptr,
        stream1
    );
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedReduce::Max(
        d_temp, temp_bytes,
        d_block_max,
        d_max_vals,
        num_proj,
        d_begin_ptr,
        d_end_ptr,
        stream1
    );
    cudaFree(d_temp);
    
    CUDA_CHECK(cudaFree(d_block_min));
    CUDA_CHECK(cudaFree(d_block_max));
}
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_flat_projection_col_idx));
    CUDA_CHECK(cudaFree(d_flat_projection_weights));

}




__global__ void FinalMinMaxKernel(
    const float* __restrict__ d_block_min,
    const float* __restrict__ d_block_max,
    float* __restrict__ d_min_vals,
    float* __restrict__ d_max_vals,
    float* __restrict__ d_bin_widths,
    const int num_blocks,
    const int num_proj,
    const int num_bins
) {
    int proj_id = blockIdx.x;
    if (proj_id >= num_proj) return;

    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;

    // sequential reduction across all blocks of this projection
    for (int i = 0; i < num_blocks; ++i) {
        int idx = proj_id * num_blocks + i;
        min_val = fminf(min_val, d_block_min[idx]);
        max_val = fmaxf(max_val, d_block_max[idx]);
    }

    d_min_vals[proj_id] = min_val;
    d_max_vals[proj_id] = max_val;
    d_bin_widths[proj_id] = (max_val > min_val)
                            ? (max_val - min_val) / (float)num_bins
                            : 1.0f;
}


// std::upper_bound not available on device. Thrust library has host-only
__device__ int binary_search_upper_bound(const float* boundaries, int num_boundaries, float val) {
    int left = 0;
    int right = num_boundaries - 1;
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (boundaries[mid] <= val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

// Add this function to generate random bin boundaries
std::vector<float> generateRandomBinBoundaries(int num_bins, int num_proj, 
                                                float* d_min_vals, float* d_max_vals) {
    // Copy min/max values to host
    std::vector<float> h_min_vals(num_proj);
    std::vector<float> h_max_vals(num_proj);
    cudaMemcpy(h_min_vals.data(), d_min_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_vals.data(), d_max_vals, num_proj * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Generate random boundaries for each projection
    std::vector<float> all_boundaries;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int p = 0; p < num_proj; ++p) {
        float min_val = h_min_vals[p];
        float max_val = h_max_vals[p];
        
        // Generate num_bins-1 random points in [0,1]
        std::vector<float> boundaries;
        boundaries.push_back(min_val);  // First boundary
        
        // Generate random points in [0,1] and scale to [min_val, max_val]
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        std::vector<float> random_points;
        for (int i = 0; i < num_bins - 1; ++i) {
            random_points.push_back(dis(gen));
        }
        
        // Sort the random points
        std::sort(random_points.begin(), random_points.end());
        
        // Scale to actual range
        for (float point : random_points) {
            boundaries.push_back(min_val + point * (max_val - min_val));
        }
        
        boundaries.push_back(max_val);  // Last boundary
        
        // Add to all_boundaries
        all_boundaries.insert(all_boundaries.end(), boundaries.begin(), boundaries.end());
    }
    
    return all_boundaries;
}


__global__ void BuildHistogramRandomWidthKernel(
    const float* __restrict__ d_attributes,
    const unsigned int* __restrict__ d_row_indices,
    const unsigned int* __restrict__ d_labels,
    // int* d_hist_class0,
    int* d_hist_class1,
    int* d_hist_class2,
    const float* d_bin_boundaries,  // array of size num_bins + 1
    const int num_rows,
    const int num_proj,
    const int num_bins
)
{
    extern __shared__ int shared_mem[];
    int proj_id = blockIdx.y;
    if (proj_id >= num_proj) return;
    
    // Initialize shared memory histograms
    for (int i = threadIdx.x; i < 3 * num_bins; i += blockDim.x)
        shared_mem[i] = 0;
    __syncthreads();
    
    // Each projection has its own set of bin boundaries
    const float* proj_boundaries = d_bin_boundaries + proj_id * (num_bins + 1);
    
    const std::size_t col_offset = std::size_t(proj_id) * num_rows;
    const int stride = blockDim.x * gridDim.x;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int i = tid; i < num_rows; i += stride) {
        float val = d_attributes[col_offset + i];
        unsigned int label = d_labels[d_row_indices[i]];
        
        // Binary search to find bin
        int bin = binary_search_upper_bound(proj_boundaries, num_bins + 1, val);
        bin = min(bin, num_bins - 1);  // Clamp to valid range
        
        // Update histogram
        if (label == 1) {
            atomicAdd(&shared_mem[num_bins + bin], 1);
        } else if (label == 2) {
            atomicAdd(&shared_mem[2 * num_bins + bin], 1);
        } else {
            atomicAdd(&shared_mem[bin], 1);
        }
    }
    __syncthreads();
    
    // Write results to global memory
    int offset = proj_id * num_bins;
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        // atomicAdd(&d_hist_class0[offset + i], shared_mem[i]);
        atomicAdd(&d_hist_class1[offset + i], shared_mem[num_bins + i]);
        atomicAdd(&d_hist_class2[offset + i], shared_mem[2 * num_bins + i]);
    }
}


template <int BLOCK_SIZE>
__global__ void BuildHistogramEqualWidthKernel(
    const float* __restrict__ d_attributes, //attributes
    const unsigned int*__restrict__ d_row_indices, //selected examples
    const unsigned int* __restrict__ d_labels,
    // int* d_hist_class0,          // [num_samples], values 0 or 1
    int* d_hist_class1,           // [NUM_BINS], count of
    int* d_hist_class2, 
    const float* d_min_vals,
    const float* d_max_vals,
    float* d_bin_widths,
    const int num_rows,
    const int num_proj,
    const int num_bins //candidate_splits 256 not 257
)
{
    extern __shared__ int shared_mem[];
    int proj_id = blockIdx.y;
    if (proj_id >= num_proj) return;
  
    // 1. Put zeroes in shared histogram
    for (int i = threadIdx.x; i < 3 * (num_bins); i += blockDim.x)
        shared_mem[i] = 0;
    __syncthreads();

    float min_val = d_min_vals[proj_id];
    float max_val = d_max_vals[proj_id];
    float bin_width = (max_val > min_val)
                            ? (max_val - min_val) / (float)(num_bins - 1)
                            : 1.0f;
    d_bin_widths[proj_id] = bin_width;

    const std::size_t col_offset = std::size_t(proj_id) * num_rows;
    const int stride = blockDim.x * gridDim.x;
    const int tid    = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < num_rows; i += stride) {
        float val   = d_attributes[col_offset + i];      
        unsigned int   label = d_labels[d_row_indices[i]];  
     
        int bin = (val >= max_val - 0.5 * bin_width) //splitting at the last threshold
                  ? (num_bins - 1)
                  : round((val - min_val) / bin_width);//have total num_bins

        if (label == 1) {
            atomicAdd(&shared_mem[num_bins + bin], 1); 
        } 
        else if (label == 2) {
            atomicAdd(&shared_mem[2 * (num_bins) + bin], 1);
        }
        else {
            atomicAdd(&shared_mem[bin], 1);
        }
    }
    __syncthreads();

    int offset =  proj_id * (num_bins);
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&d_hist_class1[offset + i], shared_mem[num_bins + i]);
        atomicAdd(&d_hist_class2[offset + i], shared_mem[2 * (num_bins) + i ]);
        // atomicAdd(&d_hist_class0[offset + i], shared_mem[i]);
    } 
}

struct index_to_proj
{
    int rows_per_proj;
    __host__ __device__
    int operator()(int idx) const { return idx / rows_per_proj; } //Linear index into "projection buckets"
};


void VariableWidthHistogram(const float* __restrict__ d_col_add_projected,
                           const unsigned int* __restrict__ d_selected_examples,
                           const unsigned int* __restrict__ d_global_labels_data,
                           float* d_min_vals,
                           float* d_max_vals,
                        //    int** d_prefix_0_out,
                           int** d_prefix_1_out,
                           int** d_prefix_2_out,
                           float** d_bin_boundaries_out,  // New output parameter
                           const int num_rows,
                           const int num_bins,
                           const int num_proj)
{
    // Generate random bin boundaries
    std::vector<float> h_boundaries = generateRandomBinBoundaries(num_bins, num_proj, 
                                                                  d_min_vals, d_max_vals);
    
    // Allocate and copy boundaries to device
    float* d_bin_boundaries;
    size_t boundaries_size = num_proj * (num_bins + 1) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_bin_boundaries, boundaries_size));
    CUDA_CHECK(cudaMemcpy(d_bin_boundaries, h_boundaries.data(), boundaries_size, 
                          cudaMemcpyHostToDevice));
    
    // Allocate histograms
    // int* d_hist_class0;
    int* d_hist_class1;
    int* d_hist_class2;
    // CUDA_CHECK(cudaMalloc(&d_hist_class0, num_proj * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_class1, num_proj * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_class2, num_proj * num_bins * sizeof(int)));
    // CUDA_CHECK(cudaMemset(d_hist_class0, 0, num_proj * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_class1, 0, num_proj * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_class2, 0, num_proj * num_bins * sizeof(int)));
    
    // Launch kernel with variable width bins
    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid, num_proj);
    int sharedMemSize = 3 * num_bins * sizeof(int);
    
    BuildHistogramRandomWidthKernel<<<grid, threads_per_block, sharedMemSize>>>(
        d_col_add_projected, d_selected_examples, d_global_labels_data,
        // d_hist_class0,
        d_hist_class1, d_hist_class2,
        d_bin_boundaries, num_rows, num_proj, num_bins);
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Prefix sums (same as EqualWidthHistogram)
    int total_rows = num_bins * num_proj;

    int* d_prefix_2;
    int* d_prefix_1;
    // int* d_prefix_0;
    CUDA_CHECK(cudaMalloc(&d_prefix_2, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_prefix_1, total_rows * sizeof(int)));
    // CUDA_CHECK(cudaMalloc(&d_prefix_0, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_prefix_2, 0, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_prefix_1, 0, total_rows * sizeof(int)));
    // CUDA_CHECK(cudaMemset(d_prefix_0, 0, total_rows * sizeof(int)));

    auto counting_begin = thrust::make_counting_iterator<int>(0);
    auto keys_begin = thrust::make_transform_iterator(counting_begin, index_to_proj{num_bins});

    auto d_hist_class2_ptr = thrust::device_pointer_cast(d_hist_class2);
    auto d_prefix_2_ptr = thrust::device_pointer_cast(d_prefix_2);

    auto d_hist_class1_ptr = thrust::device_pointer_cast(d_hist_class1);
    auto d_prefix_1_ptr = thrust::device_pointer_cast(d_prefix_1);
    
    cudaStream_t stream = 0;
 
    thrust::inclusive_scan_by_key(
        thrust::cuda::par.on(stream),
        keys_begin,
        keys_begin + total_rows,
        d_hist_class2_ptr,
        d_prefix_2_ptr,
        thrust::equal_to<int>(),
        thrust::plus<int>()
    ); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::inclusive_scan_by_key(
        thrust::cuda::par.on(stream),
        keys_begin,
        keys_begin + total_rows,
        d_hist_class1_ptr,
        d_prefix_1_ptr,
        thrust::equal_to<int>(),
        thrust::plus<int>()
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    // *d_prefix_0_out = d_prefix_0;
    *d_prefix_1_out = d_prefix_1;
    *d_prefix_2_out = d_prefix_2;
    *d_bin_boundaries_out = d_bin_boundaries;  // Return boundaries for split finding

    // CUDA_CHECK(cudaFree(d_hist_class0));
    CUDA_CHECK(cudaFree(d_hist_class1));
    CUDA_CHECK(cudaFree(d_hist_class2));
    CUDA_CHECK(cudaFree(d_max_vals));
    CUDA_CHECK(cudaFree((void *)d_selected_examples));
    CUDA_CHECK(cudaFree((void *)d_col_add_projected));
}

void VariableWidthSplit(const int* d_prefix_1,
                       const int* d_prefix_2,
                       const float* d_min_vals,
                       const float* d_bin_boundaries,
                       const int num_proj,
                       const int num_bins,
                       const int num_rows,
                       int* best_proj,
                       int* best_bin_out,
                       float* best_gain_out,
                       float* best_threshold_out,
                       int* num_pos_examples_out,
                       double* elapsed_ms,
                       const bool verbose,
                       const int comp_method)
{
    float* d_out_per_bin_per_proj;
    float* d_threshold_per_bin_per_proj;
    CUDA_CHECK(cudaMalloc(&d_out_per_bin_per_proj, num_proj * num_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_threshold_per_bin_per_proj, num_proj * num_bins * sizeof(float)));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int threads_per_block_split = 256;
    int blocks_per_grid_split = (num_bins + threads_per_block_split - 1) / threads_per_block_split;
    dim3 grid_split(blocks_per_grid_split, num_proj);
    dim3 block_split(threads_per_block_split);

    if (comp_method == 1) {
        FindBestGiniSplitVariableWidthKernel<<<grid_split, block_split>>>(
            d_prefix_1, d_prefix_2, d_bin_boundaries, num_proj, num_bins,
            d_out_per_bin_per_proj, d_threshold_per_bin_per_proj);
    }
    // Add entropy version if needed
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cub::KeyValuePair<int, float>* d_out1;
    CUDA_CHECK(cudaMalloc(&d_out1, sizeof(cub::KeyValuePair<int, float>)));
    
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins);
    
    cub::KeyValuePair<int, float> h_out1;
    CUDA_CHECK(cudaMemcpy(&h_out1, d_out1, sizeof(h_out1), cudaMemcpyDeviceToHost));

    if (h_out1.value > 0.f) {
        *best_proj = h_out1.key / num_bins;
        *best_gain_out = h_out1.value;
        *best_bin_out = h_out1.key - (*best_proj * num_bins);

        // Get the actual threshold from the boundaries
        CUDA_CHECK(cudaMemcpy(best_threshold_out, 
                             d_threshold_per_bin_per_proj + h_out1.key, 
                             sizeof(float), 
                             cudaMemcpyDeviceToHost));

        // Calculate num_pos_examples_out similar to EqualWidthSplit
        int total_count_0, total_count_1, left_count_0, left_count_1;
        CUDA_CHECK(cudaMemcpy(&total_count_0, d_prefix_1 + (*best_proj + 1) * num_bins - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&total_count_1, d_prefix_2 + (*best_proj + 1) * num_bins - 1, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&left_count_0, d_prefix_1 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&left_count_1, d_prefix_2 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost));
        *num_pos_examples_out = total_count_0 + total_count_1 - left_count_0 - left_count_1;
    }
    
    CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaFree(d_temp_storage));
    // CUDA_CHECK(cudaFree((void *)d_prefix_0));
    CUDA_CHECK(cudaFree((void *)d_prefix_1));
    CUDA_CHECK(cudaFree((void *)d_prefix_2));
    CUDA_CHECK(cudaFree((void *)d_min_vals));
    CUDA_CHECK(cudaFree((void *)d_bin_boundaries));
    CUDA_CHECK(cudaFree(d_out_per_bin_per_proj));
    CUDA_CHECK(cudaFree(d_threshold_per_bin_per_proj));
}


void EqualWidthHistogram (const float* __restrict__ d_col_add_projected, //attributes
                          const unsigned int* __restrict__ d_selected_examples, //selected examples
                          const unsigned int* __restrict__ d_global_labels_data,
                          float* d_min_vals,
                          float* d_max_vals,
                          float* d_bin_widths,
                        //   int** d_prefix_0_out,
                          int** d_prefix_1_out,
                          int** d_prefix_2_out,
                          const int num_rows, //selected_examples.size()
                          const int num_bins,
                          const int num_proj
                          )
    {

    const int BLOCK = 256;
    cudaStream_t stream = 0;

    //////////////////////////Allocate Device Memory/////////////////////////////////////////
    // auto startAlloc = std::chrono::high_resolution_clock::now();
    // int* d_hist_class0;
    int* d_hist_class1;
    int* d_hist_class2;
    // CUDA_CHECK(cudaMalloc(&d_hist_class0, num_proj * (num_bins) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_class1, num_proj * (num_bins) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_hist_class2, num_proj * (num_bins) * sizeof(int)));
    // CUDA_CHECK(cudaMemset(d_hist_class0, 0, num_proj * (num_bins) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_class1, 0, num_proj * (num_bins) * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist_class2, 0, num_proj * (num_bins) * sizeof(int)));

    
    ///////////////////////BuildHistogramEqualWidthKernel/////////////////////////////////////////
 
    //auto startHist = std::chrono::high_resolution_clock::now();
    int threads_per_block_hist = num_bins;
    //printf("threads_per_block_hist: %d\n", threads_per_block_hist);
    int num_elements_per_thread = 1;
    int blocks_per_grid_hist = (num_rows/num_elements_per_thread + threads_per_block_hist - 1) / threads_per_block_hist;
    //printf("blocks_per_grid_hist: %d\n", blocks_per_grid_hist);
    dim3 grid_hist(blocks_per_grid_hist, num_proj); //single dimension grid/projection
    //printf("grid_hist: (%d, %d)\n", grid_hist.x, grid_hist.y);
    int sharedMemSize = 3 * (num_bins) * sizeof(int); // For Hist 0, Hist 1, and Hist 2
    BuildHistogramEqualWidthKernel<BLOCK><<<grid_hist, threads_per_block_hist, sharedMemSize>>>(d_col_add_projected, d_selected_examples, d_global_labels_data,
                                                                             d_hist_class1, d_hist_class2,
                                                                             d_min_vals, d_max_vals, d_bin_widths,
                                                                             num_rows, num_proj, num_bins);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int total_rows = (num_bins) * num_proj;

    int* d_prefix_2;
    int* d_prefix_1;
    // int* d_prefix_0;
    CUDA_CHECK(cudaMalloc(&d_prefix_2, (total_rows * sizeof(int))));
    CUDA_CHECK(cudaMalloc(&d_prefix_1, (total_rows * sizeof(int))));
    // CUDA_CHECK(cudaMalloc(&d_prefix_0, (total_rows * sizeof(int))));
    CUDA_CHECK(cudaMemset(d_prefix_2, 0, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_prefix_1, 0, total_rows * sizeof(int)));
    // CUDA_CHECK(cudaMemset(d_prefix_0, 0, total_rows * sizeof(int)));

    auto counting_begin = thrust::make_counting_iterator<int>(0);
    auto keys_begin = thrust::make_transform_iterator(counting_begin, index_to_proj{num_bins});

    auto d_hist_class2_ptr   = thrust::device_pointer_cast(d_hist_class2);        // input Negative class
    auto d_prefix_2_ptr = thrust::device_pointer_cast(d_prefix_2);  // output

    auto d_hist_class1_ptr   = thrust::device_pointer_cast(d_hist_class1);        // input Positive class
    auto d_prefix_1_ptr = thrust::device_pointer_cast(d_prefix_1);  // output
    
    // auto d_hist_class0_ptr   = thrust::device_pointer_cast(d_hist_class0);        // input Not Used class
    // auto d_prefix_0_ptr = thrust::device_pointer_cast(d_prefix_0);  // output
    // cudaStream_t stream;
 
        thrust::inclusive_scan_by_key( //label = 0 second
        thrust::cuda::par.on(stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_hist_class2_ptr,                      /* values begin   */
        d_prefix_2_ptr,                    /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()
    ); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    thrust::inclusive_scan_by_key( //label = 1 first
        thrust::cuda::par.on(stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_hist_class1_ptr,                      /* values begin   */
        d_prefix_1_ptr,                /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()              /* inclusive sum  */
    );            
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());

    // *d_prefix_0_out = d_prefix_0;
    *d_prefix_1_out = d_prefix_1;
    *d_prefix_2_out = d_prefix_2;

    // CUDA_CHECK(cudaFree(d_hist_class0));
    CUDA_CHECK(cudaFree(d_hist_class1));
    CUDA_CHECK(cudaFree(d_hist_class2));
    CUDA_CHECK(cudaFree(d_max_vals));
    CUDA_CHECK(cudaFree((void *)d_selected_examples));
    CUDA_CHECK(cudaFree((void *)d_col_add_projected));
}

void BuildHistogramAndFindBestSplit(
    const float* d_col_add_projected,
    const unsigned int* d_selected_examples,
    const unsigned int* d_global_labels_data,
    float* d_min_vals,
    float* d_max_vals,
    float* d_bin_widths,
    const int num_rows,
    const int num_bins,
    const int num_proj,
    int* best_proj,
    int* best_bin_out,
    float* best_gain_out,
    float* best_threshold_out,
    int* num_pos_examples_out,
    const bool use_variable_width,  // NEW parameter
    const int comp_method)
{
    // int* d_prefix_0;
    int* d_prefix_1;
    int* d_prefix_2;
    float* d_bin_boundaries = nullptr;
    
    if (use_variable_width) {
        VariableWidthHistogram(
            d_col_add_projected, d_selected_examples, d_global_labels_data,
            d_min_vals, d_max_vals,
            // &d_prefix_0, 
            &d_prefix_1, &d_prefix_2, &d_bin_boundaries,
            num_rows, num_bins, num_proj);
            
        VariableWidthSplit(
            // d_prefix_0,
            d_prefix_1, d_prefix_2,
            d_min_vals, d_bin_boundaries,
            num_proj, num_bins, num_rows,
            best_proj, best_bin_out, best_gain_out, best_threshold_out,
            num_pos_examples_out, nullptr, false, comp_method);
    } else {
        EqualWidthHistogram(
            d_col_add_projected, d_selected_examples, d_global_labels_data,
            d_min_vals, d_max_vals, d_bin_widths,
            // &d_prefix_0,
            &d_prefix_1, &d_prefix_2,
            num_rows, num_bins, num_proj);
            
        EqualWidthSplit(
            // d_prefix_0,
            d_prefix_1, d_prefix_2,
            d_min_vals, d_bin_widths,
            num_proj, num_bins, num_rows,
            best_proj, best_bin_out, best_gain_out, best_threshold_out,
            num_pos_examples_out, nullptr, false, comp_method);
    }
}

__device__ __forceinline__
float entropy(const int pos, const int neg) {
    int total = pos + neg;
    if (total <= 0) return 0.0f;

    float p_pos = float(pos) / float(total);
    float p_neg = float(neg) / float(total);

    const float eps = 1e-30f;   // protects against denormals and log(0)

    float e = 0.0f;
    if (p_pos > eps) e -= p_pos * logf(p_pos);
    if (p_neg > eps) e -= p_neg * logf(p_neg);
    return e;
}

__device__ float gini(const int pos, const int neg) {
    int total = pos + neg;
    if (total == 0) return 0.0f;
    float p = float(pos) / total;
    float n = float(neg) / total;
    return 1.0f - p * p - n * n;
}

__global__ void FindBestGiniSplitVariableWidthKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* bin_boundaries,  // Use actual boundaries instead of min_vals/bin_widths
    const int num_proj,
    const int num_bins,
    float* gini_out_per_bin_per_proj,
    float* threshold_out_per_bin_per_proj  // Store actual thresholds
)  // per proj
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins - 1) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = hist_class0[ base_idx + num_bins - 1 ]; // last bin holds total count
    int total_class1 = hist_class1[ base_idx + num_bins - 1 ]; // last bin holds total count
    
    // Compute left class counts for this split point
    int left_class0 = hist_class0[ base_idx + bin_id ]; // hist already cumulative
    int left_class1 = hist_class1[ base_idx + bin_id ]; // hist already cumulative

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;

    float gini_left = gini(left_class1, left_class0);
    float gini_right = gini(right_class1, right_class0);
    float gini_parent = gini(total_class1, total_class0);

    float total = total_class0 + total_class1;
    float left_weight = float(left_total) / float(total);
    float right_weight = float(right_total) / float(total);

    float gini_gain = gini_parent - (left_weight * gini_left + right_weight * gini_right);

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    if (left_total == 0 || right_total == 0) {
        gini_out_per_bin_per_proj[base_idx + bin_id] = -INFINITY;
        return;
    }
    gini_out_per_bin_per_proj[base_idx + bin_id] = gini_gain;
    const float* proj_boundaries = bin_boundaries + proj_id * (num_bins + 1);
    threshold_out_per_bin_per_proj[base_idx + bin_id] = proj_boundaries[bin_id + 1];

}

__global__ void FindBestGiniSplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* min_vals,
    const float* bin_widths,
    const int num_proj,
    const int num_bins,
    float* gini_out_per_bin_per_proj
   )  // per proj
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins - 1) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = hist_class0[ base_idx + num_bins - 1 ]; // last bin holds total count
    int total_class1 = hist_class1[ base_idx + num_bins - 1 ]; // last bin holds total count
    
    // Compute left class counts for this split point
    int left_class0 = hist_class0[ base_idx + bin_id ]; // hist already cumulative
    int left_class1 = hist_class1[ base_idx + bin_id ]; // hist already cumulative

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;

    float gini_left = gini(left_class1, left_class0);
    float gini_right = gini(right_class1, right_class0);
    float gini_parent = gini(total_class1, total_class0);

    float total = total_class0 + total_class1;
    float left_weight = float(left_total) / float(total);
    float right_weight = float(right_total) / float(total);

    float gini_gain = gini_parent - (left_weight * gini_left + right_weight * gini_right);

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    if (left_total == 0 || right_total == 0) {
        gini_out_per_bin_per_proj[base_idx + bin_id] = -INFINITY;
        return;
    }
    gini_out_per_bin_per_proj[base_idx + bin_id] = gini_gain;
}

__global__ void FindBestEntropySplitKernel(
    const int* hist_class0,
    const int* hist_class1,
    const float* min_vals,
    const float* bin_widths,
    const int num_proj,
    const int num_bins,
    float* entropy_out_per_bin_per_proj
    ) 
{
    int proj_id = blockIdx.y;   
    int bin_id  = threadIdx.x + blockIdx.x * blockDim.x;

    // Only evaluate splits between bins, not after the last bin
    if (proj_id >= num_proj) return;
    if (bin_id >= num_bins) return;

    int base_idx = proj_id * num_bins;

    // Compute total class counts (redundantly across threads)
    // Total class is same for all projections
    int total_class0 = hist_class0[ (num_bins - 1) ]; // last bin holds total count
    int total_class1 = hist_class1[ (num_bins - 1) ]; // last bin holds

    // Compute left class counts for this split point
    int left_class0 = hist_class0[base_idx + bin_id];
    int left_class1 = hist_class1[base_idx + bin_id];

    int right_class0 = total_class0 - left_class0;
    int right_class1 = total_class1 - left_class1;

    int left_total  = left_class0 + left_class1;
    int right_total = right_class0 + right_class1;
    float total = total_class0 + total_class1;

    float entropy_left = entropy(left_class1, left_class0);
    float entropy_right = entropy(right_class1, right_class0);
    float entropy_before = entropy(total_class1, total_class0);

    float weighted_entropy = (left_total * entropy_left + right_total * entropy_right) / max(total, 1.0f);
    float entropy_gain = entropy_before - weighted_entropy;

    // Store per-thread result in global memory
    // just index it right so it can store the result for each bin per projection
    entropy_out_per_bin_per_proj[base_idx + bin_id] = entropy_gain;
}


void EqualWidthSplit (
    // const int* d_prefix_0,
                      const int* d_prefix_1,
                      const int* d_prefix_2,
                      const float* d_min_vals,
                      const float* d_bin_widths,
                      const int num_proj,
                      const int num_bins, //total_bins
                      const int num_rows,
                      int* best_proj,
                      int* best_bin_out,
                      float* best_gain_out,
                      float* best_threshold_out,
                      int* num_pos_examples_out,
                      double* elapsed_ms,
                      const bool verbose,
                      const int comp_method //0: entropy, 1: gini
                    )
{
    float* d_out_per_bin_per_proj;
    CUDA_CHECK(cudaMalloc(&d_out_per_bin_per_proj, num_proj * (num_bins) * sizeof(float))); // Store Entropy gain for each bin (except last)

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int threads_per_block_split = 256;
    int blocks_per_grid_split = (num_bins + threads_per_block_split - 1) / threads_per_block_split; // +1 to ensure we have enough blocks to cover all bins
    dim3 grid_split(blocks_per_grid_split, num_proj);
    dim3 block_split(threads_per_block_split);

    if (comp_method == 0) {       
        FindBestEntropySplitKernel<<<grid_split, block_split>>>(
            d_prefix_1,
            d_prefix_2, d_min_vals, d_bin_widths, num_proj, num_bins,
            d_out_per_bin_per_proj);
    }
    else {
        FindBestGiniSplitKernel<<<grid_split, block_split>>>(
            d_prefix_1, d_prefix_2, d_min_vals, d_bin_widths, num_proj, num_bins,
            d_out_per_bin_per_proj);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cub::KeyValuePair<int, float>* d_out1;
     // Allocate output
    CUDA_CHECK(cudaMalloc(&d_out1, sizeof(cub::KeyValuePair<int, float>)));
    // Step 1: Get temp storage size
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins);
    // Step 2: Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Step 3: Run ArgMax
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        d_out_per_bin_per_proj, d_out1, num_proj * num_bins
    );
    cub::KeyValuePair<int, float> h_out1;
    CUDA_CHECK(cudaMemcpy(&h_out1, d_out1, sizeof(h_out1), cudaMemcpyDeviceToHost));

    if (h_out1.value > 0.f) {
        *best_proj = h_out1.key / num_bins;
        *best_gain_out = h_out1.value;
        *best_bin_out = h_out1.key - (*best_proj * num_bins);

        int total_count_0, total_count_1, left_count_0, left_count_1;
        // Step 4: Copy result to host
        CUDA_CHECK(cudaMemcpy(&total_count_0, d_prefix_1 + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&total_count_1, d_prefix_2 + (num_bins - 1), sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&left_count_0, d_prefix_1 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&left_count_1, d_prefix_2 + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost));
        *num_pos_examples_out = total_count_0 + total_count_1 - left_count_0 - left_count_1;
        
        float min_val, bin_width;
        CUDA_CHECK(cudaMemcpy(&min_val, d_min_vals + *best_proj, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&bin_width, d_bin_widths + *best_proj, sizeof(float), cudaMemcpyDeviceToHost));
        *best_threshold_out = (*best_bin_out + 0.5) * bin_width + min_val;
    }
    
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaFree(d_temp_storage));
    // CUDA_CHECK(cudaFree((void *)d_prefix_0));
    CUDA_CHECK(cudaFree((void *)d_prefix_1));
    CUDA_CHECK(cudaFree((void *)d_prefix_2));
    CUDA_CHECK(cudaFree((void *)d_min_vals));
    CUDA_CHECK(cudaFree((void *)d_bin_widths));
    CUDA_CHECK(cudaFree(d_out_per_bin_per_proj));
}

void ThrustSortIndicesOnly(float* d_proj_values, unsigned int* d_row_ids, unsigned int* d_selected_examples, 
                            int num_rows, int num_proj) 
    {
    //Copy the list of selected example indices into a thrust::device_vector
    thrust::device_vector<unsigned int> d_selected_base(d_selected_examples, d_selected_examples + num_rows);
    //Obtain a raw device pointer to that internal storage
    const unsigned int* d_base_ptr =
        thrust::raw_pointer_cast(d_selected_base.data());
    auto d_row_ids_iter = thrust::device_pointer_cast(d_row_ids);

    thrust::transform(
        thrust::make_counting_iterator<int>(0), // first  input iterator
        thrust::make_counting_iterator<int>(num_rows * num_proj), // last  iterator
        d_row_ids_iter, // output iterator
        [=] __device__ (int i) { // device lambda
            return d_base_ptr[i % num_rows];  // replicate indices
    });

    //auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < num_proj; ++k)
    {
        const int offset = k * num_rows;
        // keys  : projection values
        thrust::device_ptr<float>        keys_begin (d_proj_values + offset);
        // values: the row indices we want to permute with the keys
        thrust::device_ptr<unsigned int> vals_begin (d_row_ids_iter + offset);

        // stable_sort the keys; move the indices with them
        thrust::stable_sort_by_key(keys_begin,
                                   keys_begin + num_rows,
                                   vals_begin);
    }
    cudaFree((void *)d_selected_examples);
}

__device__ __forceinline__
float atomicMaxFloat(float* addr, float v)
{
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), v)));
    } while (assumed != old);
    return __int_as_float(old);
}

template <typename T>
struct ArgMaxPair {
    __device__ __forceinline__
    cub::KeyValuePair<T,int> operator()(
        const cub::KeyValuePair<T,int>& a,
        const cub::KeyValuePair<T,int>& b) const 
    {
        if (a.key > b.key) return a;
        if (b.key > a.key) return b;
        // tie-breaker: larger index wins
        return (a.value > b.value) ? a : b;
    }
};

template<int STRIDE, int blockSize>
__global__ void GiniGainKernel(
    const int* __restrict__ d_prefix_pos,      // [num_proj * proj_size / STRIDE]
    const int* __restrict__ d_prefix_neg,      // [num_proj * proj_size / STRIDE]
    const float* __restrict__ d_values,
    int* d_block_best_split,                   // [num_proj * num_blocks]
    float* d_block_best_gain,                  // [num_proj * num_blocks]
    const int num_rows, const int logical_rows)
{

    using Pair       = cub::KeyValuePair<float,int>;
    using BlockReduce = cub::BlockReduce<Pair, blockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;


    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int proj_id = blockIdx.y;
    int base = proj_id * num_rows;

    int total_pos = d_prefix_pos[base + num_rows - 1];
    int total_neg = d_prefix_neg[base + num_rows - 1];
    
    int logical_start = block_id * blockSize;
    int logical_end   = min(logical_start + blockSize, logical_rows); // last block check with min

    const int logical_row = logical_start + tid;  // global logical row this thread would handle
    float gain = -INFINITY;
    int split = -1;

    // Handle degenerate case (uniform across block → safe to return)
    if (num_rows < 2 || logical_rows <= 0) {
        if (tid == 0) {
            int out_idx = proj_id * gridDim.x + block_id;
            d_block_best_gain[out_idx]  = -INFINITY;
            d_block_best_split[out_idx] = -1;
        }
        return;  // all threads take this, so no block-wide ops after
    }

    if (logical_row < logical_end && logical_row < logical_rows) {
        // compute candidate split index using STRIDE mapping
        int idx = (logical_row + 1) * STRIDE - 1;
        //idx = min(idx, num_rows - 2); // ensure idx+1 valid
        // if (idx > num_rows - 2) idx = num_rows - 2;
        // if (idx < 0) idx = 0;

        // only consider split if value actually changes between idx and idx+1
        if (idx >= 0 && idx < num_rows - 1) {
            if (d_values[base + idx] != d_values[base + idx + 1]) {
                int left_pos = d_prefix_pos[base + idx];
                int left_neg = d_prefix_neg[base + idx];
                int right_pos = total_pos - left_pos;
                int right_neg = total_neg - left_neg;
                float left_g = gini(left_pos, left_neg);
                float right_g = gini(right_pos, right_neg);
                float parent_g = gini(total_pos, total_neg);

                int total = left_pos + left_neg + right_pos + right_neg;
                float left_weight = float(left_pos + left_neg) / float(total);
                float right_weight = float(right_pos + right_neg) / float(total);

                gain = parent_g - (left_weight * left_g + right_weight * right_g);
                split = idx + 1;
            }
        }
    }
    Pair thread_pair(gain, split);

    // Custom ArgMax: if gains tie, pick one deterministically (e.g., larger split)
    // Block-wide reduction
    Pair block_best = BlockReduce(temp_storage).Reduce(thread_pair, ArgMaxPair<float>());

    if (tid == 0) {
        int out_idx = proj_id * gridDim.x + block_id;
        d_block_best_gain [out_idx] = block_best.key;
        d_block_best_split[out_idx] = block_best.value;
    }
}

template<int STRIDE, int blockSize>
__global__ void EntropyGainKernel(
    const int* __restrict__ d_prefix_pos,      // [num_proj * proj_size / STRIDE]
    const int* __restrict__ d_prefix_neg,      // [num_proj * proj_size / STRIDE]
    const float* __restrict__ d_values,
    int* d_block_best_split,                   // [num_proj * num_blocks]
    float* d_block_best_gain,                  // [num_proj * num_blocks]
    const int num_rows, const int logical_rows)
{

    using Pair       = cub::KeyValuePair<float,int>;
    using BlockReduce = cub::BlockReduce<Pair, blockSize>;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int proj_id = blockIdx.y;

    int base = proj_id * num_rows;

    int total_pos = d_prefix_pos[base + num_rows - 1]; //Last arrary value per projection == total pos
    int total_neg = d_prefix_neg[base + num_rows - 1];

    const int logical_start = block_id * blockSize;
    const int logical_end   = min(logical_start + blockSize, logical_rows); // exclusive

    const int logical_row = logical_start + tid;  // global logical row this thread would handle
    float gain = -INFINITY;
    int split = -1;

    // Handle degenerate case (uniform across block → safe to return)
    if (num_rows < 2 || logical_rows <= 0) {
        if (tid == 0) {
            int out_idx = proj_id * gridDim.x + block_id;
            d_block_best_gain[out_idx]  = -INFINITY;
            d_block_best_split[out_idx] = -1;
        }
        return;  // all threads take this, so no block-wide ops after
    }

    if (logical_row < logical_end && logical_row < logical_rows) {
        // compute candidate split index using STRIDE mapping
        int idx = (logical_row + 1) * STRIDE - 1;
        //idx = min(idx, num_rows - 2); // ensure idx+1 valid
        // if (idx > num_rows - 2) idx = num_rows - 2;
        // if (idx < 0) idx = 0;

        // only consider split if value actually changes between idx and idx+1
        if (idx >= 0 && idx < num_rows - 1) {
            if (d_values[base + idx] < d_values[base + idx + 1]) {
                int left_pos  = d_prefix_pos[base + idx];
                int left_neg  = d_prefix_neg[base + idx];
                int right_pos = total_pos - left_pos;
                int right_neg = total_neg - left_neg;

                int left_count  = left_pos + left_neg;
                int right_count = right_pos + right_neg;        
                int total_count = left_count + right_count;

                float parent_e = entropy(total_pos, total_neg);
                float left_e = entropy(left_pos, left_neg);
                float right_e = entropy(right_pos, right_neg);

                // weight by counts; use num_rows (per-proj) to normalize — keep as float
                gain = parent_e - (left_count * left_e + right_count * right_e) / float(total_count);
                split = idx + 1;
            }
        }
    }
    Pair thread_pair(gain, split);

    // Custom ArgMax: if gains tie, pick one deterministically (e.g., larger split)
    // Block-wide reduction
    Pair block_best = BlockReduce(temp_storage).Reduce(thread_pair, ArgMaxPair<float>());

    if (tid == 0) {
        int out_idx = proj_id * gridDim.x + block_id;
        d_block_best_gain [out_idx] = block_best.key;
        d_block_best_split[out_idx] = block_best.value;
    }
}

__global__ void buildPosFlag(const unsigned int* sorted_idx,
                             const unsigned int* labels,
                             int*                flag,
                             int                 N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
        flag[tid] = (labels[ sorted_idx[tid] ] == 1) ? 1 : 0;
}

/* 0 ↔ 1  (turn positives into negatives or vice-versa) */
__global__ void invertFlag(int* flag, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) flag[tid] = 1 - flag[tid];
}


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
    const bool verbose,
    const int comp_method
    )
{
    CUDA_CHECK(cudaGetLastError()); 
    constexpr int STRIDE = 1; // You can adjust STRIDE for # of elements for split computation
    const int blockSize = 256;

    const int logical_rows = num_rows - 1;//(num_rows + STRIDE - 1) / STRIDE;

    int total_rows = num_proj * num_rows;
    int* d_prefix_pos;
    int* d_prefix_neg;
    CUDA_CHECK(cudaMalloc(&d_prefix_pos, (total_rows * sizeof(int) )));
    CUDA_CHECK(cudaMalloc(&d_prefix_neg, (total_rows * sizeof(int) )));
    CUDA_CHECK(cudaMemset(d_prefix_pos, 0, total_rows * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_prefix_neg, 0, total_rows * sizeof(int)));

    int dimX = (logical_rows + blockSize - 1) / blockSize;
    size_t shared_mem = sizeof(float) * blockSize + sizeof(int) * blockSize;
    int total_blocks = num_proj * dimX;
    int* d_block_best_split;
    float* d_block_best_gain;
    CUDA_CHECK(cudaMalloc(&d_block_best_split, total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_best_gain, total_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_block_best_split, 0xFF, total_blocks * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_block_best_gain, 0xFF, total_blocks * sizeof(float)));
  
    
    auto startPrefixThrust = std::chrono::high_resolution_clock::now();
    int* d_flag;
    cudaStream_t stream = 0;
    CUDA_CHECK(cudaMalloc(&d_flag, total_rows * sizeof(int)));
    dim3 gridCUB((total_rows + blockSize - 1) / blockSize);//1D grid
    buildPosFlag<<<gridCUB, blockSize, 0, stream>>>(d_sorted_indices,
                                             d_labels,
                                             d_flag,
                                             total_rows);
    // cudaDeviceSynchronize();
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Make Thrust iterators */
    auto counting_begin = thrust::make_counting_iterator<int>(0);
    auto keys_begin =
            thrust::make_transform_iterator(counting_begin,
                                            index_to_proj{num_rows});
    auto d_flag_ptr   = thrust::device_pointer_cast(d_flag);        // input
    auto d_prefix_pos_ptr = thrust::device_pointer_cast(d_prefix_pos);  // output
    auto d_prefix_neg_ptr = thrust::device_pointer_cast(d_prefix_neg);  // output

    thrust::inclusive_scan_by_key( //label = 1 first
        thrust::cuda::par.on(stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_flag_ptr,                      /* values begin   */
        d_prefix_pos_ptr,                /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()              /* inclusive sum  */
    );            
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());


    invertFlag<<<gridCUB, blockSize, 0, stream>>>(d_flag, total_rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::inclusive_scan_by_key( //label = 0 second
        thrust::cuda::par.on(stream),
        keys_begin,                      /* keys   begin   */
        keys_begin + total_rows,         /* keys   end     */
        d_flag_ptr,                      /* values begin   */
        d_prefix_neg_ptr,                    /* output         */
        thrust::equal_to<int>(),         /* identical keys */
        thrust::plus<int>()
    ); 
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_flag));

    dim3 gridDim(dimX, num_proj); //int dimX = (logical_rows + blockSize - 1) / blockSize;
    if (comp_method == 0) {
        EntropyGainKernel<STRIDE, blockSize><<<gridDim, blockSize, shared_mem>>>(
        d_prefix_pos, d_prefix_neg, d_col_add_projected,
        d_block_best_split, d_block_best_gain, num_rows, logical_rows);
    } else {    
        GiniGainKernel<STRIDE, blockSize><<<gridDim, blockSize, shared_mem>>>(
        d_prefix_pos, d_prefix_neg, d_col_add_projected,
        d_block_best_split, d_block_best_gain, num_rows, logical_rows);
    }
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    void* d_temp_storage1 = nullptr;
    size_t temp_storage_bytes1 = 0;

    cub::KeyValuePair<int, float>* d_out1;
     // Allocate output
    CUDA_CHECK(cudaMalloc(&d_out1, sizeof(cub::KeyValuePair<int, float>)));
        // Step 1: Get temp storage size
    cub::DeviceReduce::ArgMax(
        d_temp_storage1, temp_storage_bytes1,
        d_block_best_gain, d_out1, total_blocks);
    // Step 2: Allocate temp storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage1, temp_storage_bytes1));
    // Step 3: Run ArgMax
    cub::DeviceReduce::ArgMax(
        d_temp_storage1, temp_storage_bytes1,
        d_block_best_gain, d_out1, total_blocks
    );
    cub::KeyValuePair<int, float> h_out1;
    CUDA_CHECK(cudaMemcpy(&h_out1, d_out1, sizeof(h_out1), cudaMemcpyDeviceToHost));

    if (h_out1.value > 0.f) {
        *best_gain_out = h_out1.value;
        CUDA_CHECK(cudaMemcpy(best_split_out, d_block_best_split + h_out1.key, sizeof(int), cudaMemcpyDeviceToHost));
        *best_proj = h_out1.key / dimX;
        float threshold_1, threshold_2;
        int proj_offset = (*best_proj) * num_rows;
        CUDA_CHECK(cudaMemcpy(&threshold_1, d_col_add_projected + proj_offset + (*best_split_out) - 1, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&threshold_2, d_col_add_projected + proj_offset + (*best_split_out), sizeof(float), cudaMemcpyDeviceToHost));
        *best_threshold_out = 0.5f * (threshold_1 + threshold_2);
    }

    CUDA_CHECK(cudaFree(d_out1));
    CUDA_CHECK(cudaFree(d_temp_storage1));
    CUDA_CHECK(cudaFree(d_prefix_pos));
    CUDA_CHECK(cudaFree(d_prefix_neg));
    CUDA_CHECK(cudaFree(d_block_best_split));
    CUDA_CHECK(cudaFree(d_block_best_gain));
    CUDA_CHECK(cudaFree(d_col_add_projected));
    CUDA_CHECK(cudaFree(d_sorted_indices));
}
