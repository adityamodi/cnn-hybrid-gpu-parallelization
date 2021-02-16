// #include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// #include <cuda_runtime.h>
// #include <device_launch_parameters.h>

// #include <cublas_v2.h>
// #include <cudnn.h>

// using namespace std;

int main(){
	int proc_id = 1;
	int len;
	int num_samp;
	std::string fname = "mnist_training_images_" + std::to_string(proc_id) + ".csv";
	std::ifstream in_file (fname);
	std::vector<unsigned int> data;
	std::vector<unsigned int> labels;
	if(in_file.is_open()){
		in_file >> num_samp >> len;
		std::cout << num_samp << len << "\n";
		for(int line = 0; line < num_samp; line++){
			for(int feat = 0; feat < len; feat++){
				unsigned int x;
				// std::cout << x;
				in_file >> x;
				data.push_back(x);
			}
		}
	}
	in_file.close();
	fname = "mnist_training_labels_" + std::to_string(proc_id) + ".csv";
	std::ifstream in_file2 (fname);
	if(in_file2.is_open()){
		in_file2 >> num_samp;
		for(int line = 0; line < num_samp; line++){
				unsigned int x;
				in_file2 >> x;
				labels.push_back(x);
		}
	}
	for(int i=0; i < 10; i++){
		std::cout << "Image " << i << ":\n";
		for(int j=0; j < len; j++) std::cout << data[len*i + j] << " ";
		std::cout << "\nLabel: " << labels[i] << "\n";
	}
	return 0;
}