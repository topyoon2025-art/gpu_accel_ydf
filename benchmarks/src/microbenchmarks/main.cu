// main.cu
#include "randomprojection.cuh"  // Put your code in a header file

// Generate trunk dataset
void generateTrunkData(std::vector<float>& h_data, 
                      std::vector<unsigned int>& h_labels,
                      int num_rows, int num_features, unsigned int seed) {
    const int kNInformative = 256;
    const int ninform = std::min(kNInformative, num_features);
    
    // Pre-compute the two means for the informative coordinates
    std::vector<float> mu0(num_features, 0.0f);
    std::vector<float> mu1(num_features, 0.0f);
    
    for (int j = 0; j < ninform; ++j) {
        const float f = 1.0f / std::sqrt(static_cast<float>(j + 1));
        mu0[j] = -f;
        mu1[j] = f;
    }
    
    // Fill the feature columns
    for (int j = 0; j < num_features; ++j) {
        // Deterministic per-column seed
        std::seed_seq seq{seed, static_cast<unsigned int>(j)};
        std::mt19937 gen(seq);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        
        for (int i = 0; i < num_rows; ++i) {
            const bool cls1 = (i >= num_rows / 2);
            const float mean = cls1 ? mu1[j] : mu0[j];
            h_data[i * num_features + j] = mean + normal(gen);
        }
    }
    
    // Fill the label column
    for (int i = 0; i < num_rows; ++i) {
        h_labels[i] = (i >= num_rows / 2) ? 1 : 0;  // 0-based labels for binary classification
    }
}

// Create a simple dataset where we know the optimal split
void generateSimpleTestData(std::vector<float>& h_data, 
                           std::vector<unsigned int>& h_labels,
                           int num_rows, int num_features) {
    // Create a simple 2D dataset where feature 0 perfectly separates classes
    // Class 0: feature 0 in range [-2, -1]
    // Class 1: feature 0 in range [1, 2]
    
    for (int i = 0; i < num_rows; ++i) {
        bool is_class1 = (i >= num_rows / 2);
        h_labels[i] = is_class1 ? 1 : 0;
        
        for (int j = 0; j < num_features; ++j) {
            if (j == 0) {
                // Feature 0: perfectly separates classes
                if (is_class1) {
                    h_data[i * num_features + j] = 1.0f + (float)(i - num_rows/2) / (float)(num_rows/2);
                } else {
                    h_data[i * num_features + j] = -2.0f + (float)i / (float)(num_rows/2);
                }
            } else {
                // Other features: random noise
                h_data[i * num_features + j] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            }
        }
    }
}

void printUsage(const char* program_name) {
    printf("Usage: %s [--toy | --trunk]\n", program_name);
    printf("  --toy    Use simple toy dataset (default)\n");
    printf("  --trunk  Use trunk synthetic dataset\n");
}

// // Trunk main
// int main() {
//     // Initialize CUDA
//     warmupfunction();
    
//     // Test parameters
//     const int num_rows = 100000;
//     const int num_features = 100;
//     const int num_proj = 10;
//     const int num_bins = 256;
//     const int num_selected = 50000;  // subset of rows
//     const unsigned int seed = 42;  // Fixed seed for reproducibility
    
//     // Generate test data on host
//     std::vector<float> h_data(num_features * num_rows);
//     std::vector<unsigned int> h_labels(num_rows);
//     std::vector<unsigned int> h_selected_examples(num_selected);
    
//     // Use Trunk data generator
//     printf("Generating Trunk dataset with %d rows and %d features...\n", num_rows, num_features);
//     generateTrunkData(h_data, h_labels, num_rows, num_features, seed);
    
//     // Select random subset of examples
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::vector<int> indices(num_rows);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), gen);
//     for (int i = 0; i < num_selected; ++i) {
//         h_selected_examples[i] = indices[i];
//     }
    
//     // Generate random projections
//     std::vector<std::vector<int>> projection_col_idx(num_proj);
//     std::vector<std::vector<float>> projection_weights(num_proj);
    
//     std::uniform_int_distribution<int> feat_dis(0, num_features - 1);
//     std::uniform_real_distribution<float> weight_dis(-1.0f, 1.0f);
    
//     for (int p = 0; p < num_proj; ++p) {
//         int num_features_per_proj = 5 + (p % 5);  // 5-10 features per projection
//         for (int f = 0; f < num_features_per_proj; ++f) {
//             projection_col_idx[p].push_back(feat_dis(gen));
//             projection_weights[p].push_back(weight_dis(gen));
//         }
//     }
    
//     // Print some statistics about the generated data
//     printf("\nDataset statistics:\n");
//     int class0_count = 0, class1_count = 0;
//     for (int i = 0; i < num_rows; ++i) {
//         if (h_labels[i] == 0) class0_count++;
//         else class1_count++;
//     }
//     printf("  Class 0: %d samples\n", class0_count);
//     printf("  Class 1: %d samples\n", class1_count);
    
//     // Compute and print mean/std for first few features
//     printf("\nFeature statistics (first 5 features):\n");
//     for (int j = 0; j < std::min(5, num_features); ++j) {
//         float sum0 = 0, sum1 = 0, sum_sq0 = 0, sum_sq1 = 0;
//         int count0 = 0, count1 = 0;
        
//         for (int i = 0; i < num_rows; ++i) {
//             float val = h_data[i * num_features + j];
//             if (h_labels[i] == 0) {
//                 sum0 += val;
//                 sum_sq0 += val * val;
//                 count0++;
//             } else {
//                 sum1 += val;
//                 sum_sq1 += val * val;
//                 count1++;
//             }
//         }
        
//         float mean0 = sum0 / count0;
//         float mean1 = sum1 / count1;
//         float std0 = std::sqrt(sum_sq0 / count0 - mean0 * mean0);
//         float std1 = std::sqrt(sum_sq1 / count1 - mean1 * mean1);
        
//         printf("  Feature %d: Class 0 (mean=%.3f, std=%.3f), Class 1 (mean=%.3f, std=%.3f)\n",
//                j, mean0, std0, mean1, std1);
//     }
    
//     // Allocate device memory
//     float* d_data;
//     unsigned int* d_labels;
//     unsigned int* d_selected_examples;
//     float* d_col_add_projected;
    
//     CUDA_CHECK(cudaMalloc(&d_data, num_features * num_rows * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_labels, num_rows * sizeof(unsigned int)));
//     CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
//     CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
    
//     // Copy data to device
//     CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_features * num_rows * sizeof(float), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
//     // Test equal-width histogram
//     printf("\nTesting Equal-Width Histogram...\n");
//     auto start_equal = std::chrono::high_resolution_clock::now();
    
//     float* d_min_vals_eq = nullptr;
//     float* d_max_vals_eq = nullptr;
//     float* d_bin_widths_eq = nullptr;
//     double elapsed_apply_eq = 0;
    
//     // Apply projection with equal-width (split_method = 2)
//     ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
//                             &d_min_vals_eq, &d_max_vals_eq, &d_bin_widths_eq,
//                             projection_col_idx, projection_weights,
//                             num_selected, num_proj, num_rows,
//                             &elapsed_apply_eq, 2, true);
    
//     // Build histogram and find best split
//     int* d_prefix_0_eq, *d_prefix_1_eq, *d_prefix_2_eq;
//     EqualWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
//                        d_min_vals_eq, d_max_vals_eq, d_bin_widths_eq,
//                        &d_prefix_0_eq, &d_prefix_1_eq, &d_prefix_2_eq,
//                        num_selected, num_bins, num_proj);
    
//     int best_proj_eq, best_bin_eq, num_pos_examples_eq;
//     float best_gain_eq, best_threshold_eq;
//     double elapsed_split_eq = 0;
    
//     EqualWidthSplit(d_prefix_0_eq, d_prefix_1_eq, d_prefix_2_eq,
//                    d_min_vals_eq, d_bin_widths_eq,
//                    num_proj, num_bins, num_selected,
//                    &best_proj_eq, &best_bin_eq, &best_gain_eq, &best_threshold_eq,
//                    &num_pos_examples_eq, &elapsed_split_eq, true, 1);  // 1 = gini
    
//     auto end_equal = std::chrono::high_resolution_clock::now();
    
//     // Test variable-width histogram
//     printf("\nTesting Variable-Width Histogram...\n");
    
//     // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
//     CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
//     CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
//     auto start_var = std::chrono::high_resolution_clock::now();
    
//     float* d_min_vals_var = nullptr;
//     float* d_max_vals_var = nullptr;
//     float* d_bin_widths_var = nullptr;
//     double elapsed_apply_var = 0;
    
//     // Apply projection with variable-width (split_method = 3)
//     ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
//                             &d_min_vals_var, &d_max_vals_var, &d_bin_widths_var,
//                             projection_col_idx, projection_weights,
//                             num_selected, num_proj, num_rows,
//                             &elapsed_apply_var, 3, true);
    
//     // Build variable-width histogram and find best split
//     int* d_prefix_0_var, *d_prefix_1_var, *d_prefix_2_var;
//     float* d_bin_boundaries_var;
//     VariableWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
//                           d_min_vals_var, d_max_vals_var,
//                           &d_prefix_0_var, &d_prefix_1_var, &d_prefix_2_var,
//                           &d_bin_boundaries_var,
//                           num_selected, num_bins, num_proj);
    
//     int best_proj_var, best_bin_var, num_pos_examples_var;
//     float best_gain_var, best_threshold_var;
//     double elapsed_split_var = 0;
    
//     VariableWidthSplit(d_prefix_0_var, d_prefix_1_var, d_prefix_2_var,
//                       d_min_vals_var, d_bin_boundaries_var,
//                       num_proj, num_bins, num_selected,
//                       &best_proj_var, &best_bin_var, &best_gain_var, &best_threshold_var,
//                       &num_pos_examples_var, &elapsed_split_var, true, 1);  // 1 = gini
    
//     auto end_var = std::chrono::high_resolution_clock::now();
    
//     // Print results
//     printf("\n=== RESULTS ===\n");
//     printf("Equal-Width Histogram:\n");
//     printf("  Best projection: %d\n", best_proj_eq);
//     printf("  Best bin: %d\n", best_bin_eq);
//     printf("  Best gain: %f\n", best_gain_eq);
//     printf("  Best threshold: %f\n", best_threshold_eq);
//     printf("  Total time: %f ms\n", 
//            std::chrono::duration<double, std::milli>(end_equal - start_equal).count());
    
//     printf("\nVariable-Width Histogram:\n");
//     printf("  Best projection: %d\n", best_proj_var);
//     printf("  Best bin: %d\n", best_bin_var);
//     printf("  Best gain: %f\n", best_gain_var);
//     printf("  Best threshold: %f\n", best_threshold_var);
//     printf("  Total time: %f ms\n", 
//            std::chrono::duration<double, std::milli>(end_var - start_var).count());
    
//     printf("\nSpeedup (Equal/Variable): %.2fx\n", 
//            std::chrono::duration<double, std::milli>(end_var - start_var).count() /
//            std::chrono::duration<double, std::milli>(end_equal - start_equal).count());
    
//     // Cleanup
//     CUDA_CHECK(cudaFree(d_data));
//     CUDA_CHECK(cudaFree(d_labels));
    
//     return 0;
// }


// toy main
int main() {
    // Initialize CUDA
    warmupfunction();
    
    // Small test parameters
    const int num_rows = 20;  // Small enough to manually verify
    const int num_features = 5;
    const int num_proj = 3;
    const int num_bins = 10;  // Small number of bins
    const int num_selected = num_rows;  // Use all rows
    
    // Generate simple test data
    std::vector<float> h_data(num_features * num_rows);
    std::vector<unsigned int> h_labels(num_rows);
    std::vector<unsigned int> h_selected_examples(num_selected);
    
    printf("=== SIMPLE TEST DATASET ===\n");
    generateSimpleTestData(h_data, h_labels, num_rows, num_features);
    
    // Select all examples
    for (int i = 0; i < num_selected; ++i) {
        h_selected_examples[i] = i;
    }
    
    // Print the dataset
    printf("\nDataset (first feature only, which should separate classes):\n");
    printf("Row\tFeature0\tLabel\n");
    for (int i = 0; i < num_rows; ++i) {
        printf("%d\t%.3f\t\t%d\n", i, h_data[i * num_features], h_labels[i]);
    }
    
    // Create simple projections
    std::vector<std::vector<int>> projection_col_idx(num_proj);
    std::vector<std::vector<float>> projection_weights(num_proj);
    
    // Projection 0: Just feature 0 (should find perfect split around 0)
    projection_col_idx[0] = {0};
    projection_weights[0] = {1.0f};
    
    // Projection 1: Feature 0 + noise
    projection_col_idx[1] = {0, 1};
    projection_weights[1] = {1.0f, 0.1f};
    
    // Projection 2: Only noise features
    projection_col_idx[2] = {1, 2};
    projection_weights[2] = {1.0f, 1.0f};
    
    printf("\nProjections:\n");
    for (int p = 0; p < num_proj; ++p) {
        printf("Projection %d: features [", p);
        for (int f = 0; f < projection_col_idx[p].size(); ++f) {
            printf("%d", projection_col_idx[p][f]);
            if (f < projection_col_idx[p].size() - 1) printf(", ");
        }
        printf("] with weights [");
        for (int f = 0; f < projection_weights[p].size(); ++f) {
            printf("%.2f", projection_weights[p][f]);
            if (f < projection_weights[p].size() - 1) printf(", ");
        }
        printf("]\n");
    }
    
    // Compute expected best split
    printf("\n=== EXPECTED RESULTS ===\n");
    printf("For projection 0 (just feature 0):\n");
    printf("  Class 0 range: [-2.0, -1.0]\n");
    printf("  Class 1 range: [1.0, 2.0]\n");
    printf("  Optimal split threshold: ~0.0 (anywhere between -1.0 and 1.0)\n");
    printf("  Expected Gini gain: 0.5 (perfect split from 0.5 impurity to 0.0)\n");
    
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
    
    // Test 1: Exact Split
    printf("\n=== TEST 1: EXACT SPLIT ===\n");
    
    // Apply projection for exact method
    float* d_col_add_projected_exact;
    unsigned int* d_selected_examples_exact;
    CUDA_CHECK(cudaMalloc(&d_col_add_projected_exact, num_proj * num_selected * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples_exact, num_selected * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples_exact, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    double elapsed_apply_exact = 0;
    ApplyProjectionColumnADD(d_data, d_selected_examples_exact, d_col_add_projected_exact,
                            nullptr, nullptr, nullptr,
                            projection_col_idx, projection_weights,
                            num_selected, num_proj, num_rows,
                            &elapsed_apply_exact, 0, true);  // 0 = no histogram
    
    // Sort and find exact split
    unsigned int* d_sorted_indices;
    CUDA_CHECK(cudaMalloc(&d_sorted_indices, num_proj * num_rows * sizeof(unsigned int)));
    
    ThrustSortIndicesOnly(d_col_add_projected_exact, d_sorted_indices, 
                         d_selected_examples_exact, num_rows, num_proj);
    
    float best_gain_exact = -INFINITY;
    int best_split_exact = -1;
    float best_threshold_exact = 0;
    int best_proj_exact = -1;
    double elapsed_exact = 0;
    
    ExactSplit(d_sorted_indices, d_labels,
              &best_gain_exact, &best_split_exact, &best_threshold_exact, &best_proj_exact,
              num_rows, num_proj, d_col_add_projected_exact,
              &elapsed_exact, true, 1);  // 1 = gini
    
    printf("Best projection: %d\n", best_proj_exact);
    printf("Best split index: %d\n", best_split_exact);
    printf("Best gain: %f\n", best_gain_exact);
    printf("Best threshold: %f\n", best_threshold_exact);
    
    // Test 2: Equal-Width Histogram
    printf("\n=== TEST 2: EQUAL-WIDTH HISTOGRAM ===\n");
    
    // Reallocate
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    float* d_min_vals_eq = nullptr;
    float* d_max_vals_eq = nullptr;
    float* d_bin_widths_eq = nullptr;
    double elapsed_apply_eq = 0;
    
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_eq, &d_max_vals_eq, &d_bin_widths_eq,
                            projection_col_idx, projection_weights,
                            num_selected, num_proj, num_rows,
                            &elapsed_apply_eq, 2, true);  // 2 = equal-width
    
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
    
    printf("Best projection: %d\n", best_proj_eq);
    printf("Best bin: %d\n", best_bin_eq);
    printf("Best gain: %f\n", best_gain_eq);
    printf("Best threshold: %f\n", best_threshold_eq);
    
    // Test 3: Variable-Width Histogram
    printf("\n=== TEST 3: VARIABLE-WIDTH HISTOGRAM ===\n");
    
    // Reallocate
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_selected * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_selected * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_selected * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    float* d_min_vals_var = nullptr;
    float* d_max_vals_var = nullptr;
    float* d_bin_widths_var = nullptr;
    double elapsed_apply_var = 0;
    
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_var, &d_max_vals_var, &d_bin_widths_var,
                            projection_col_idx, projection_weights,
                            num_selected, num_proj, num_rows,
                            &elapsed_apply_var, 3, true);  // 3 = variable-width
    
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
    
    printf("Best projection: %d\n", best_proj_var);
    printf("Best bin: %d\n", best_bin_var);
    printf("Best gain: %f\n", best_gain_var);
    printf("Best threshold: %f\n", best_threshold_var);
    
    // Compare results
    printf("\n=== COMPARISON ===\n");
    printf("Method\t\tBest Proj\tBest Gain\tBest Threshold\n");
    printf("Exact\t\t%d\t\t%.4f\t\t%.4f\n", best_proj_exact, best_gain_exact, best_threshold_exact);
    printf("Equal-Width\t%d\t\t%.4f\t\t%.4f\n", best_proj_eq, best_gain_eq, best_threshold_eq);
    printf("Variable-Width\t%d\t\t%.4f\t\t%.4f\n", best_proj_var, best_gain_var, best_threshold_var);
    
    printf("\nExpected: Projection 0 should have best gain (~0.5) with threshold around 0.0\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    
    return 0;
}