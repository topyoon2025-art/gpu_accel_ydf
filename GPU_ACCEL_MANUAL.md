# System setup
Create development environment for GPU acceleration  

## System to build on
Ubuntu 24.04 in AWS g6.4xlarge or General  

## Repository update and upgrade: 
sudo apt update && sudo apt upgrade -y  

## Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3.sh  
bash Miniconda3.sh  
conda create -n ydf-accel python=3.13  

## Restart terminal for miniconda to be effective
Need to restart terminal to use miniconda for the first time  

## Activate miniconda (base) and (ydf-accel)
conda activate  
conda activate ydf-accel  

## Create appropriate directory
mkdir projects  

## Now, clone gpu_accel_ydf in projects directory
git clone https://github.com/topyoon2025-art/gpu_accel_ydf.git  

## Bazel installation
sudo curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazel   
chmod +x /usr/local/bin/bazel   
bazel version (to check version to be 6.5.0)  
example    
bazel build //examples:train_oblique_forest    
bazel-bin/examples/train_oblique_forest --input_mode csv --max_num_projections 100 --num_trees 1 --label_col target --numerical_split_type 'Equal Width' --num_threads 1 --tree_depth 2 --train_csv /home/ubuntu/projects/dataset/1048576x100.csv  

sudo reboot  

## Install gcc and g++ 12 as gcc/g++ 13 not compatible with the latest CUDA Toolkit
sudo apt update  
sudo apt install gcc-12 g++-12  

## Install CUDA ToolKit: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb  
sudo dpkg -i cuda-keyring_1.1-1_all.deb  
sudo apt-get update  
sudo apt-get -y install cuda-toolkit-13-1  

## Install Nvidia Driver, choose one below and you can switch between
sudo apt-get install -y nvidia-open  
sudo apt-get install -y cuda-drivers  

## Set environment variables for cuda-13.1
export CUDA_HOME=/usr/local/cuda-13.1  
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.1  
export PATH=$CUDA_HOME/bin:$PATH  
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  

## Set symlink for cuda-13.1 to generic path
sudo ln -sfn /usr/local/cuda-13.1 /usr/local/cuda  

## Set up treeple and panda for python file to generate dataset
python -m pip install treeple   
pip install pandas  

## Files changed/modified from https://github.com/ariellubonja/yggdrasil-oblique-forests.git
		modified:   .bazelrc
        modified:   .vscode/c_cpp_properties.json
        new file:   1000x1000.csv
        modified:   WORKSPACE
        new file:   cuda-keyring_1.1-1_all.deb
        modified:   examples/BUILD
        modified:   examples/train_oblique_forest.cc
        new file:   generate_trunk_data.py
        modified:   yggdrasil_decision_forests/dataset/BUILD
        modified:   yggdrasil_decision_forests/dataset/vertical_dataset_io.cc
        modified:   yggdrasil_decision_forests/dataset/vertical_dataset_io.h
        modified:   yggdrasil_decision_forests/learner/decision_tree/BUILD
        modified:   yggdrasil_decision_forests/learner/decision_tree/oblique.cc
        new file:   yggdrasil_decision_forests/learner/decision_tree/randomprojection.cu
        new file:   yggdrasil_decision_forests/learner/decision_tree/randomprojection.hpp

## References to functions in randomprojection.cu and oblique.cc

--oblique.cc  	
	--Set use_GPU variable to 1 to use GPU and set it to 0 to use CPU  
	--Prepare all projection using the below to break out of for loop  
		std::vector<std::vector<int>> projection_col_idx;//Stores column indices per projection for GPU function  
		std::vector<std::vector<float>> projection_weights;//Stores weights per column per projection for GPU function  
		std::vector<Projection> current_projections;  
	--Only SampleProjection in for loop  
	--Copy selected examples indices to device   
	--Create d_min_vals, d_max_vals, d_bin_widths to be available in this file  
	--ApplyProjectionColumnAdd  
		--Perform Apply Projections  
		--Get min vals, max vals and bin widths for Equal Width and Random Histogram  
	--RandomHistogram  
		--Random Histogram binning  
	--EqaualWidthHistogram  
		--Equal Histogram binning  
	--HistogramSplit  
		--Split for both Random and Equal Width Histogram  
			--Computes Best Bin, Best Gain, Best Threshold, and Num Pos Examples  
	--ThrustSortIndicesOnly  
		--Sort the indices from selected examples based on values   
	--ExactSplit  
		--Split for Exact   
	--Update best_condition in YDF for the next iterations  
			
--randomprojection.cu  
	--ColumnAddProjectionKernel  
		--Device kernel to compute Apply Projection  
	--ColumnAddComputeMinMaxCombined  
		--Device kernel to compute Apply Projection / min and max values for Histogram  
	--ApplyProjectionColumnADD  
		--Host Function   
		--Prep with weights and offsets  
			--col_per_proj: Number of columns per projection since SampleProjection produces different number of columns for projection //lambda function to get size of each inner vector  
			--offset: Get the offset using inclusive scan since it will be flattened out of vector of vectors  
			--Total Size: Calculate total size for flattening of the vector of vectors using accumulate  
			--Copy and Flatten projection data structures using memcpy for both column indices and weights  
			--Copy offset, column indices, and weiths to device  
		--Call ColumnAddProjectionKernel for Exact  
		--Call ColumnAddComputeMinMaxCombined for Histogram binning for both Equal Width and Random binning  
		--Retrieve min and max values from each block  
			--gridDim.x is the number of blocks per projection and d_block_min and d_block_max contains the min and max values per block  
			--Retrieve min and max values per the number of blocks per projection using offsets.  
	--BuildHistogramEqualWidthKernel  	
		--Device Kernel for Equal Width Histogram binning for 3 classes, cutomized to 2 classes 0 and 1  
		--Arthmetic binning by (max - min) / (num_bins)  
	--struct index_to_proj  
		--Callable object, Linear index into "projection buckets", number of bins per projection  
	--EqualWidthHistogram  
		--Host Function for Equal Width Histogram  
		--Calls BuildHistogramEqualWidthKernel  
		--Compute prefix sums by doing inclusive scan by key  
	--lower_bound_naive_device  
		--Device kernel for Random binning  
		--Compare all bin boundaries until matching with the right bin then return bin number  
	--BuildHistogramRandomKernel  
		--Device Kernel for Random binning  
		--Two Options  
			--lower_bound_naive_device explained above  
			--cub::lowerbound for binning by binary search tree method  
	--RandomHistogram  
		--Copy min and max vals for projections from device to host  
		--Generate random splits for all projections  
		--Sort the random split boundaries per segment / per the number of bins  
		--Call BuildHistogramRandomKernel  
		--Compute prefix sums by doing inclusive scan by key  
	--entropy  
		--Device kernel to compute Entropy  
	--gini  
		--Device kernel to compute Gini  
	--FindBestGiniSplitKernel  
		--Device kernel to compute gini out per bin per projection  
	--FindBestEntropySplitKernel  
		--Device kernel to compute entropy out per bin per projection  
	--HistogramSplit  
		--Host Function  
		--Call FindBestEntropySplitKernel or FindBestGiniSplitKernel based on the compute method  
		--Retrieve max using cub::DeviceReduce::ArgMax  
		--Get Best Projection, Best Gain, Best Bin, Num Pos Examples, Best Threshold  
		--Need just gain as the array is 1 D and index information is not lost since not reduced block wide.  Possible since we have a relatively small number of bins per projection.  
	--ThrustSortIndicesOnly  
		--Host Function  
		--Replicate the selected example indices for each projection  
		--Sort the indices based on the values per projection  
	--atomicMaxFloat  
		--Performs an atomic max on a float  
	--struct ArgMaxPair  
		--Return (value, index) pair with the maximum value, Key is gain and value is split  
	--GiniGainKernel  
		--Can do equal frequency quantiles  
		--Device kernel  
		--Compute the best gini gain and split per block through block-wide reduction  
		--Then thread 0 puts the max value and index per block  
	--EntropyGainKernel  
		--Can do equal frequency quantiles  
		--Device kernel  
		--Compute the best gini gain and split per block through block-wide reduction  
		--Then thread 0 puts the max value and index per block  
	--buildPosFlag  
		--Device kernel in 1D grid  
		--Assign labels of 1  
	--invertFlag  
		--Device kernel to turn positives into negatives  
		--Assign labels of 0  
	--ExactSplit  	
		--Host Function 
		--Call buildPosFlag to assign positive labels then perform inclusive scan by key for each projection to compute prefix sums for positive labels  
		--Call invertFlag to assign negative labels then perform inclusive scan by key for each projection to compute prefix sums for negative labels  
		--Call EntropyGainKernel or GiniGainKernel to get max value and index per block based on comp method  
		--Retrieve max gain and key out of all the blocks  
		--Get Best Projection, Best Gain, Best Threshold  
		--Needed both gain and split so it can be reduced block wide  
		





