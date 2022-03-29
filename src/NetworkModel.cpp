#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>

#include "NetworkModel.h"

/*
std::vector<std::string> stringSplit(const std::string &str, const std::string &delim) {
	std::regex reg(delim);
	std::vector<std::string> sub_string(std::sregex_token_iterator(str.begin(), str.end(), reg, -1), 
										std::sregex_token_iterator());
	return sub_string;
}*/

NetworkModel::NetworkModel(std::string model_name): model_name(model_name) {
	// get information from the file
	std::string dir = "model/Models/" + model_name + "/";

	std::ifstream fin_dim(dir + "dimension.txt");
	//std::cout << dir + "dimension.txt" << std::endl;
	int num_dim, total_neuron = 0;
	fin_dim >> num_dim;

	dim.resize(num_dim);
	for (int i = 0; i < num_dim; ++ i) {
		fin_dim >> dim[i];
		total_neuron += dim[i];
	}
	fin_dim.close();
	
	// small scale
	if (num_dim <= 3 && dim[0] <= 256 && num_dim >= 2 && dim[1] <= 64 && num_dim >= 3 && dim[2] <= 16) {
		weight.resize(2);
		std::ifstream fin_w1(dir + "weights1.txt");
		weight[0].resize(dim[0]);
		for (int i = 0; i < dim[0]; ++ i) {
			weight[0][i].resize(dim[1]);
			for (int j = 0; j < dim[1]; ++ j) {
				fin_w1 >> weight[0][i][j];
				//std::cout << weight[0][i][j] << " ";
			}
			//std::cout << std::endl;
		}
		fin_w1.close();

		std::ifstream fin_w2(dir + "weights2.txt");
		weight[1].resize(dim[1]);
		for (int i = 0; i < dim[1]; ++ i) {
			weight[1][i].resize(dim[2]);
			for (int j = 0; j < dim[2]; ++ j)
				fin_w2 >> weight[1][i][j];
		}
		fin_w2.close();
	}
	//large scale
	else {
		std::ifstream fin_w(dir + "weights.txt");
		neuron_list.reserve(total_neuron);
		std::string read_buffer;
		for (int i = 0; i < total_neuron; ++ i) {
			getline(fin_w, read_buffer);
		}
	}
}

std::vector <std::vector <std::vector <float> > >& NetworkModel::getWeight() {
	return weight;
}

NetworkInput::NetworkInput(std::string dataset_name) {
	// initialize an input
	fin_input.open("model/Datasets/" + dataset_name + "/input.txt", std::ifstream::in);
	fin_output.open("model/Datasets/" + dataset_name + "/output.txt", std::ifstream::in);

	std::ifstream fin_info("model/Datasets/" + dataset_name + "/info.txt");
	fin_info >> input_dim;
	fin_info >> input_size;
	fin_info.close();
	input_matrix.resize(input_dim);
}

NetworkInput::~NetworkInput() {
	fin_input.close();
	fin_output.close();
}

std::vector <float>& NetworkInput::getInputMatrix() {
	//std::cout << input_dim << std::endl;
	for (int i = 0; i < input_dim; ++ i) {
		fin_input >> input_matrix[i];
		//std::cout << input_matrix[i];
	}
	//std::cout << std::endl;
	return input_matrix;
}

int NetworkInput::getOutput() {
	int output_val;
	fin_output >> output_val;
	//std::cout << "Read Value: " << output_val << std::endl;
	return output_val;
}


std::pair <int, int> NetworkInput::getInputInfo() {
	return std::make_pair(input_dim, input_size);
}
