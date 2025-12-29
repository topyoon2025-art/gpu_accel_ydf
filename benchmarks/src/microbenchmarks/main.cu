//TODO E.W. bins may not be matching YDF. they seem to do some midpoint computation

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
            h_data[j*num_rows + i] = mean + normal(gen);
        }
    }
    
    // Fill the label column
    for (int i = 0; i < num_rows; ++i) {
        h_labels[i] = (i >= num_rows / 2) ? 2 : 1;  // 0-based labels for binary classification
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
        h_labels[i] = is_class1 ? 2 : 1;
        
        for (int j = 0; j < num_features; ++j) {
            if (j == 0) {
                // Feature 0: perfectly separates classes
                if (is_class1) {
                    h_data[j*num_rows + i] = 1.0f + (float)(i - num_rows/2) / (float)(num_rows/2);
                } else {
                    h_data[j*num_rows + i] = -2.0f + (float)i / (float)(num_rows/2);
                }
            } else {
                // Other features: random noise
                h_data[j*num_rows + i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            }
        }
    }
}

void printUsage(const char* program_name) {
    printf("Usage: %s [--toy | --trunk [num_rows]]\n", program_name);
    printf("  --toy              Use simple toy dataset (default)\n");
    printf("  --trunk [num_rows] Use trunk synthetic dataset with specified number of rows (default: 100000)\n");
}

// Trunk main
int main(int argc, char** argv) {
    // Parse command line arguments
    bool use_trunk = true; // Default to trunk
    int trunk_num_rows = 100000; // Default number of rows for trunk dataset
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--toy") == 0) {
            use_trunk = false;
        } else if (strcmp(argv[i], "--trunk") == 0) {
            use_trunk = true;
            // Check if next argument is a number (num_rows)
            if (i + 1 < argc) {
                char* endptr;
                long val = strtol(argv[i + 1], &endptr, 10);
                if (*endptr == '\0' && val > 0) {
                    trunk_num_rows = (int)val;
                    i++; // Skip the next argument since we've processed it
                }
            }
        } else {
            printUsage(argv[0]);
            return 1;
        }
    }

    // Initialize CUDA
    warmupfunction();
    
    // Test parameters
    int num_rows, num_features, num_proj, num_bins;
    const unsigned int seed = 42;  // Fixed seed for reproducibility

    if (use_trunk) {
        num_rows = trunk_num_rows;
        num_features = 100; // TODO reduced from 4096 due to GPU memory limitations.
        num_proj = 75;//sqrt(num_features) * 1.5;
        num_bins = 1024;
        // num_rows = num_rows;
    } else {
        // Toy dataset parameters
        num_rows = 20;           // Small enough to manually verify
        num_features = 3;
        num_proj = 300;
        num_bins = 10;           // Small number of bins
        num_rows = num_rows; // Use all rows
    }
    
    // Generate test data on host
    std::vector<float> h_data(num_features * num_rows);
    std::vector<unsigned int> h_labels(num_rows);
    std::vector<unsigned int> h_selected_examples(num_rows);
    
    // Generate Data based on flag
    if (use_trunk) {
        printf("Generating Trunk dataset with %d rows and %d features...\n", num_rows, num_features);
        generateTrunkData(h_data, h_labels, num_rows, num_features, seed);
    } else {
        printf("Generating Toy dataset with %d rows and %d features...\n", num_rows, num_features);
        generateSimpleTestData(h_data, h_labels, num_rows, num_features);

        // Print the whole of the 1st feature (index 0)
        printf("\n--- 1st Feature Values (Toy Dataset) ---\n");
        printf("%-10s | %-15s | %-10s\n", "Row Index", "Feature 0 Val", "Label");
        printf("--------------------------------------------\n");
        for (int i = 0; i < num_rows; ++i) {
            float val = h_data[0 * num_rows + i]; // Feature index 0
            unsigned int label = h_labels[i];
            printf("%-10d | %-15.6f | %-10u\n", i, val, label);
        }
        printf("--------------------------------------------\n\n");
    }
    
    // Select random subset of examples
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> indices(num_rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < num_rows; ++i) {
        h_selected_examples[i] = indices[i];
    }
    
    // Generate random projections
    std::vector<std::vector<int>> projection_col_idx(num_proj);
    std::vector<std::vector<float>> projection_weights(num_proj);

    // 0) Deterministic projection: feature 0 with weight 1
    projection_col_idx[0] = {0};
    projection_weights[0] = {1.0f};
    
    // Use a fixed seed for reproducibility
    std::mt19937 gen_proj(12345);
    std::uniform_int_distribution<int> feat_dis(0, num_features - 1);
    std::uniform_real_distribution<float> weight_dis(-1.0f, 1.0f);

    // 1..num_proj-1 random projections
    for (int p = 1; p < num_proj; ++p) {
        int num_features_per_proj = 5 + (p % 5);  // 5-9 features per projection
        projection_col_idx[p].reserve(num_features_per_proj);
        projection_weights[p].reserve(num_features_per_proj);
        for (int f = 0; f < num_features_per_proj; ++f) {
            projection_col_idx[p].push_back(feat_dis(gen_proj));
            projection_weights[p].push_back(weight_dis(gen_proj));
        }
    }
    
    // Print some statistics about the generated data
    printf("\nDataset statistics:\n");
    int class0_count = 0, class1_count = 0, class2_count = 0;
    for (int i = 0; i < num_rows; ++i) {
        if (h_labels[i] == 0) class0_count++;
        else if (h_labels[i] == 1) class1_count++;
        else if (h_labels[i] == 2) class2_count++;
    }
    printf("  Class 0: %d samples (reserved by YDF)\n", class0_count);
    printf("  Class 1: %d samples\n", class1_count);
    printf("  Class 2: %d samples\n", class2_count);
    
    // Compute and print mean/std for first few features
    printf("\nFeature statistics (first 5 features):\n");
    for (int j = 0; j < std::min(5, num_features); ++j) {
        float sum0 = 0, sum1 = 0, sum_sq0 = 0, sum_sq1 = 0;
        int count0 = 0, count1 = 0;
        
        for (int i = 0; i < num_rows; ++i) {
            float val = h_data[j*num_rows + i];
            if (h_labels[i] == 1) {
                sum0 += val;
                sum_sq0 += val * val;
                count0++;
            } else if (h_labels[i] == 2) {
                sum1 += val;
                sum_sq1 += val * val;
                count1++;
            }
        }
        
        float mean0 = count0 > 0 ? sum0 / count0 : 0.0f;
        float mean1 = count1 > 0 ? sum1 / count1 : 0.0f;
        float std0 = count0 > 0 ? std::sqrt(std::max(0.0f, sum_sq0 / count0 - mean0 * mean0)) : 0.0f;
        float std1 = count1 > 0 ? std::sqrt(std::max(0.0f, sum_sq1 / count1 - mean1 * mean1)) : 0.0f;
        
        printf("  Feature %d: Class 1 (mean=%.3f, std=%.3f), Class 2 (mean=%.3f, std=%.3f)\n",
               j, mean0, std0, mean1, std1);
    }
    
    // Allocate device memory
    float* d_data;
    unsigned int* d_labels;
    unsigned int* d_selected_examples;
    float* d_col_add_projected;
    
    CUDA_CHECK(cudaMalloc(&d_data, num_features * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_rows * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), num_features * num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Test equal-width histogram
    printf("\nTesting Equal-Width Histogram...\n");
    auto start_equal = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_eq = nullptr;
    float* d_max_vals_eq = nullptr;
    float* d_bin_widths_eq = nullptr;
    double elapsed_apply_eq = 0;
    
    // Apply projection with equal-width (split_method = 2)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_eq, &d_max_vals_eq, &d_bin_widths_eq,
                            projection_col_idx, projection_weights,
                            num_rows, num_proj, num_rows,
                            &elapsed_apply_eq, 2, true);
    
    // Build histogram and find best split
    int* d_prefix_0_eq, *d_prefix_1_eq, *d_prefix_2_eq;
    EqualWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                       d_min_vals_eq, d_max_vals_eq, d_bin_widths_eq,
                    //    &d_prefix_0_eq,
                       &d_prefix_1_eq, &d_prefix_2_eq,
                       num_rows, num_bins, num_proj);
    
    int best_proj_eq, best_bin_eq, num_pos_examples_eq;
    float best_gain_eq, best_threshold_eq;
    double elapsed_split_eq = 0;
    
    EqualWidthSplit(
        // d_prefix_0_eq,
        d_prefix_1_eq, d_prefix_2_eq,
                   d_min_vals_eq, d_bin_widths_eq,
                   num_proj, num_bins, num_rows,
                   &best_proj_eq, &best_bin_eq, &best_gain_eq, &best_threshold_eq,
                   &num_pos_examples_eq, &elapsed_split_eq, true, 1);  // 1 = gini
    
    auto end_equal = std::chrono::high_resolution_clock::now();

    // Test variable-width histogram
    printf("\nTesting Binary Search Variable-Width Histogram...\n");
    
    // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    auto start_var_binary_search = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_var_binary_search = nullptr;
    float* d_max_vals_var_binary_search = nullptr;
    float* d_bin_widths_var_binary_search = nullptr;
    double elapsed_apply_var_binary_search = 0;
    
    // Apply projection with variable-width (split_method = 3)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_var_binary_search, &d_max_vals_var_binary_search, &d_bin_widths_var_binary_search,
                            projection_col_idx, projection_weights,
                            num_rows, num_proj, num_rows,
                            &elapsed_apply_var_binary_search, 3, true);
    
    // Build variable-width histogram and find best split
    int* d_prefix_0_var_binary_search, *d_prefix_1_var_binary_search, *d_prefix_2_var_binary_search;
    float* d_bin_boundaries_var_binary_search;
    VariableWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                          d_min_vals_var_binary_search, d_max_vals_var_binary_search,
                        //   &d_prefix_0_var,
                          &d_prefix_1_var_binary_search, &d_prefix_2_var_binary_search,
                          &d_bin_boundaries_var_binary_search,
                          num_rows, num_bins, num_proj, VBIN_BINARY);
    
    int best_proj_var_binary_search, best_bin_var_binary_search, num_pos_examples_var_binary_search;
    float best_gain_var_binary_search, best_threshold_var_binary_search;
    double elapsed_split_var_binary_search = 0;
    
    VariableWidthSplit(
        // d_prefix_0_var,
        d_prefix_1_var_binary_search, d_prefix_2_var_binary_search,
                      d_min_vals_var_binary_search, d_bin_boundaries_var_binary_search,
                      num_proj, num_bins, num_rows,
                      &best_proj_var_binary_search, &best_bin_var_binary_search, &best_gain_var_binary_search, &best_threshold_var_binary_search,
                      &num_pos_examples_var_binary_search, &elapsed_split_var_binary_search, true, 1);  // 1 = gini
    
    auto end_var_binary_search = std::chrono::high_resolution_clock::now();
    
    // Test variable-width histogram
    printf("\nTesting Linear Scan Variable-Width Histogram...\n");
    
    // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    auto start_var_linear_scan = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_var_linear_scan = nullptr;
    float* d_max_vals_var_linear_scan = nullptr;
    float* d_bin_widths_var_linear_scan = nullptr;
    double elapsed_apply_var_linear_scan = 0;
    
    // Apply projection with variable-width (split_method = 3)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_var_linear_scan, &d_max_vals_var_linear_scan, &d_bin_widths_var_linear_scan,
                            projection_col_idx, projection_weights,
                            num_rows, num_proj, num_rows,
                            &elapsed_apply_var_linear_scan, 3, true);
    
    // Build variable-width histogram and find best split
    int* d_prefix_0_var_linear_scan, *d_prefix_1_var_linear_scan, *d_prefix_2_var_linear_scan;
    float* d_bin_boundaries_var_linear_scan;
    VariableWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                          d_min_vals_var_linear_scan, d_max_vals_var_linear_scan,
                        //   &d_prefix_0_var,
                          &d_prefix_1_var_linear_scan, &d_prefix_2_var_linear_scan,
                          &d_bin_boundaries_var_linear_scan,
                          num_rows, num_bins, num_proj, VBIN_LINEAR);
    
    int best_proj_var_linear_scan, best_bin_var_linear_scan, num_pos_examples_var_linear_scan;
    float best_gain_var_linear_scan, best_threshold_var_linear_scan;
    double elapsed_split_var_linear_scan = 0;
    
    VariableWidthSplit(
        // d_prefix_0_var,
        d_prefix_1_var_linear_scan, d_prefix_2_var_linear_scan,
                      d_min_vals_var_linear_scan, d_bin_boundaries_var_linear_scan,
                      num_proj, num_bins, num_rows,
                      &best_proj_var_linear_scan, &best_bin_var_linear_scan, &best_gain_var_linear_scan, &best_threshold_var_linear_scan,
                      &num_pos_examples_var_linear_scan, &elapsed_split_var_linear_scan, true, 1);  // 1 = gini
    
    auto end_var_linear_scan = std::chrono::high_resolution_clock::now();


    printf("\nTesting Variable-Width (Binary Search Method) Histogram...\n");
    
    // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    auto start_var_2_pass = std::chrono::high_resolution_clock::now();
    
    float* d_min_vals_var_2_pass = nullptr;
    float* d_max_vals_var_2_pass = nullptr;
    float* d_bin_widths_var_2_pass = nullptr;
    double elapsed_apply_var_2_pass = 0;
    
    // Apply projection with variable-width (split_method = 3)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_var_2_pass, &d_max_vals_var_2_pass, &d_bin_widths_var_2_pass,
                            projection_col_idx, projection_weights,
                            num_rows, num_proj, num_rows,
                            &elapsed_apply_var_2_pass, 3, true);
    
    // Build variable-width histogram and find best split
    int* d_prefix_0_var_2_pass, *d_prefix_1_var_2_pass, *d_prefix_2_var_2_pass;
    float* d_bin_boundaries_var_2_pass;
    VariableWidthHistogram(d_col_add_projected, d_selected_examples, d_labels,
                          d_min_vals_var_2_pass, d_max_vals_var_2_pass,
                        //   &d_prefix_0_var,
                          &d_prefix_1_var_2_pass, &d_prefix_2_var_2_pass,
                          &d_bin_boundaries_var_2_pass,
                          num_rows, num_bins, num_proj, VBIN_2_PASS);
    
    int best_proj_var_2_pass, best_bin_var_2_pass, num_pos_examples_var_2_pass;
    float best_gain_var_2_pass, best_threshold_var_2_pass;
    double elapsed_split_var_2_pass = 0;
    
    VariableWidthSplit(
        // d_prefix_0_var,
        d_prefix_1_var_2_pass, d_prefix_2_var_2_pass,
                      d_min_vals_var_2_pass, d_bin_boundaries_var_2_pass,
                      num_proj, num_bins, num_rows,
                      &best_proj_var_2_pass, &best_bin_var_2_pass, &best_gain_var_2_pass, &best_threshold_var_2_pass,
                      &num_pos_examples_var_2_pass, &elapsed_split_var_2_pass, true, 1);  // 1 = gini
    
    auto end_var_2_pass = std::chrono::high_resolution_clock::now();
    
    // Test exact splitting
    printf("\nTesting Exact Split...\n");

    // Need to reallocate d_col_add_projected and d_selected_examples as they were freed
    CUDA_CHECK(cudaMalloc(&d_col_add_projected, num_proj * num_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_selected_examples, num_rows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_selected_examples, h_selected_examples.data(), num_rows * sizeof(unsigned int), cudaMemcpyHostToDevice));

    auto start_exact = std::chrono::high_resolution_clock::now();

    float* d_min_vals_exact = nullptr;
    float* d_max_vals_exact = nullptr;
    float* d_bin_widths_exact = nullptr;
    double elapsed_apply_exact = 0;

    // Apply projection (split_method = 0 for exact)
    ApplyProjectionColumnADD(d_data, d_selected_examples, d_col_add_projected,
                            &d_min_vals_exact, &d_max_vals_exact, &d_bin_widths_exact,
                            projection_col_idx, projection_weights,
                            num_rows, num_proj, num_rows,
                            &elapsed_apply_exact, 0, true);  // 0 = exact split method

    // Allocate memory for sorted indices
    unsigned int* d_sorted_indices;
    CUDA_CHECK(cudaMalloc(&d_sorted_indices, num_proj * num_rows * sizeof(unsigned int)));

    // Sort indices (required for exact split)
    ThrustSortIndicesOnly(d_col_add_projected, d_sorted_indices, d_selected_examples, 
                        num_rows, num_proj);

    // Perform exact split
    int best_proj_exact, best_split_exact;
    float best_gain_exact, best_threshold_exact;
    double elapsed_split_exact = 0;

    ExactSplit(d_sorted_indices, d_labels, 
            &best_gain_exact, &best_split_exact, &best_threshold_exact,
            &best_proj_exact,
            num_rows, num_proj, d_col_add_projected,
            &elapsed_split_exact, true, 1);  // 1 = gini

    auto end_exact = std::chrono::high_resolution_clock::now();

    // Print results
    printf("\n=== RESULTS ===\n");

    printf("\nExact Split:\n");
    printf("  Best projection: %d\n", best_proj_exact);
    printf("  Best split index: %d\n", best_split_exact);
    printf("  Best gain: %f\n", best_gain_exact);
    printf("  Best threshold: %f\n", best_threshold_exact);
    printf("  Total time: %f ms\n", 
    std::chrono::duration<double, std::milli>(end_exact - start_exact).count());

    printf("\nEqual-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_eq);
    printf("  Best bin: %d\n", best_bin_eq);
    printf("  Best gain: %f\n", best_gain_eq);
    printf("  Best threshold: %f\n", best_threshold_eq);
    double time_equal = std::chrono::duration<double, std::milli>(end_equal - start_equal).count();
    printf("  Total time: %f ms\n", time_equal);
           
    printf("\nBinary Search Variable-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_var_binary_search);
    printf("  Best bin: %d\n", best_bin_var_binary_search);
    printf("  Best gain: %f\n", best_gain_var_binary_search);
    printf("  Best threshold: %f\n", best_threshold_var_binary_search);
    double time_var_binary_search = std::chrono::duration<double, std::milli>(end_var_binary_search - start_var_binary_search).count();
    printf("  Total time: %f ms\n", time_var_binary_search);

    printf("\nLinear Scan Variable-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_var_linear_scan);
    printf("  Best bin: %d\n", best_bin_var_linear_scan);
    printf("  Best gain: %f\n", best_gain_var_linear_scan);
    printf("  Best threshold: %f\n", best_threshold_var_linear_scan);
    double time_var_linear_scan = std::chrono::duration<double, std::milli>(end_var_linear_scan - start_var_linear_scan).count();
    printf("  Total time: %f ms\n", time_var_linear_scan);

    printf("\n2-pass Variable-Width Histogram:\n");
    printf("  Best projection: %d\n", best_proj_var_2_pass);
    printf("  Best bin: %d\n", best_bin_var_2_pass);
    printf("  Best gain: %f\n", best_gain_var_2_pass);
    printf("  Best threshold: %f\n", best_threshold_var_2_pass);
    double time_var_2_pass = std::chrono::duration<double, std::milli>(end_var_2_pass - start_var_2_pass).count();
    printf("  Total time: %f ms\n", time_var_2_pass);
    
    // printf("\nTiming Comparison:\n");
    
    // double time_exact = std::chrono::duration<double, std::milli>(end_exact - start_exact).count();

    // printf("  Speedup (Variable/Exact): %.2fx\n", time_exact / time_var);
    // printf("  Speedup (Variable/Equal): %.2fx\n", time_equal / time_var);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_labels));
    
    return 0;
}