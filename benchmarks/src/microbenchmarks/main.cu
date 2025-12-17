// main.cu
#include "randomprojection.cuh"  // Put your code in a header file

int main() {
    // Initialize CUDA
    warmupfunction();
    
    // Test parameters
    const int num_rows = 10000;
    const int num_features = 100;
    const int num_proj = 10;
    const int num_bins = 256;
    const int num_selected = 5000;  // subset of rows
    
    // Generate test data on host
    std::vector<float> h_data(num_features * num_rows);
    std::vector<unsigned int> h_labels(num_rows);
    std::vector<unsigned int> h_selected_examples(num_selected);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    std::uniform_int_distribution<unsigned int> label_dis(0, 1);
    
    for (int i = 0; i < num_features * num_rows; ++i) {
        h_data[i] = dis(gen);
    }
    
    for (int i = 0; i < num_rows; ++i) {
        h_labels[i] = label_dis(gen);
    }
    
    // Select random subset of examples
    std::vector<int> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < num_selected; ++i) {
        h_selected_examples[i] = indices[i];
    }
    
    // Generate random projections
    std::vector<std::vector<int>> projection_col_idx(num_proj);
    std::vector<std::vector<float>> projection_weights(num_proj);
    
    std::uniform_int_distribution<int> feat_dis(0, num_features - 1);
    std::uniform_real_distribution<float> weight_dis(-1.0f, 1.0f);
    
    for (int p = 0; p < num_proj; ++p) {
        int num_features_per_proj = 5 + (p % 5);  // 5-10 features per projection
        for (int f = 0; f < num_features_per_proj; ++f) {
            projection_col_idx[p].push_back(feat_dis(gen));
            projection_weights[p].push_back(weight_dis(gen));
        }
    }
    
    // Allocate device memory
    float* d_data;
    unsigned int* d_labels;
    unsigned int* d_selected_examples;
    float* d_col_add_projected;
    
    CUDA_CHECK(cudaMalloc(&d_data, num_features * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_features * num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Test equal-width histogram
    printf("Testing Equal-Width Histogram...\n");
    auto start_equal = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_eq = nullptr;
    float* d_max_vals_eq = nullptr;
    float* d_bin_widths_eq = nullptr;
    double elapsed_apply_eq = 0;
    
    // Apply projection with equal-width (split_method = 2)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_eq, &d_max_vals_eq, &d_bin_widths_eq,
                            projection_col_idx, projection_weights,
                            num_selected, num_proj, num_rows,
                            &elapsed_apply_eq, 2, true);
    
    // Build histogram and find best split
    int* d_prefix_0_eq, *d_prefix_1_eq, *d_prefix_2_eq;
    EqualWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                       d_min_vals_eq, d_max_vals_eq, d_bin_widths_eq,
                       &d_prefix_0_eq, &d_prefix_1_eq, &d_prefix_2_eq,
                       num_selected, num_bins, num_proj);
    
    int best_proj_eq, best_bin_eq, num_pos_examples_eq;
    float best_gain_eq, best_threshold_eq;
    double elapsed_split_eq = 0;
    
    EqualWidthSplit(d_prefix_0_eq, d_prefix_1_eq, d_prefix_2_eq,
                   d_min_vals_eq, d_bin_widths_eq,
                   num_proj, num_bins, num_selected,
                   &best_proj_eq, &best_bin_eq, &best_gain_eq, &best_threshold_eq,
                   &num_pos_examples_eq, &elapsed_split_eq, true, 1);  // 1 = gini
    
    auto end_equal = std::chrono::high_resolution_clock::now();
    
    // Test variable-width histogram
    printf("\nTesting Variable-Width Histogram...\n");
    
    // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    auto start_var = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_var = nullptr;
    float* d_max_vals_var = nullptr;
    float* d_bin_widths_var = nullptr;
    double elapsed_apply_var = 0;
    
    // Apply projection with variable-width (split_method = 3)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_var, &d_max_vals_var, &d_bin_widths_var,
                            projection_col_idx, projection_weights,
                            num_selected, num_proj, num_rows,
                            &elapsed_apply_var, 3, true);
    
    // Build variable-width histogram and find best split
    int* d_prefix_0_var, *d_prefix_1_var, *d_prefix_2_var;
    float* d_bin_boundaries_var;
    VariableWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                          d_min_vals_var, d_max_vals_var,
                          &d_prefix_0_var, &d_prefix_1_var, &d_prefix_2_var,
                          &d_bin_boundaries_var,
                          num_selected, num_bins, num_proj);
    
    int best_proj_var, best_bin_var, num_pos_examples_var;
    float best_gain_var, best_threshold_var;
    double elapsed_split_var = 0;
    
    VariableWidthSplit(d_prefix_0_var, d_prefix_1_var, d_prefix_2_var,
                      d_min_vals_var, d_bin_boundaries_var,
                      num_proj, num_bins, num_selected,
                      &best_proj_var, &best_bin_var, &best_gain_var, &best_threshold_var,
                      &num_pos_examples_var, &elapsed_split_var, true, 1);  // 1 = gini
    
    auto end_var = std::chrono::high_resolution_clock::now();
    
    // Print results
    printf("\n=== RESULTS ===\n");
    printf("Equal-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_eq);
    printf("  Best bin: %d\n", best_bin_eq);
    printf("  Best gain: %f\n", best_gain_eq);
    printf("  Best threshold: %f\n", best_threshold_eq);
    printf("  Total time: %f ms\n", 
           std::chrono::duration<double, std::milli>(end_equal - start_equal).count());
    
    printf("\nVariable-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_var);
    printf("  Best bin: %d\n", best_bin_var);
    printf("  Best gain: %f\n", best_gain_var);
    printf("  Best threshold: %f\n", best_threshold_var);
    printf("  Total time: %f ms\n", 
           std::chrono::duration<double, std::milli>(end_var - start_var).count());
    
    printf("\nSpeedup (Equal/Variable): %.2fx\n", 
           std::chrono::duration<double, std::milli>(end_var - start_var).count() /
           std::chrono::duration<double, std::milli>(end_equal - start_equal).count());
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    
    return 0;
}