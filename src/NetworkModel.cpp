#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>

#include "NetworkModel.h"

NetworkModel::NetworkModel(std::string &model_name): model_name(model_name) {
	// get information from the file
	std::string dir = "model/Models/" + model_name + "/";

	dim.resize(3);
	std::ifstream fin_dim(dir + "dimension.txt");
	//std::cout << dir + "dimension.txt" << std::endl;
	for (int i = 0; i < 3; ++ i)
		fin_dim >> dim[i];
	fin_dim.close();

	weight.resize(2);
	std::ifstream fin_w1(dir + "weights1.txt");
	weight[0].resize(dim[0]);
	for (int i = 0; i < dim[0]; ++ i) {
		weight[0][i].resize(dim[1]);
		for (int j = 0; j < dim[1]; ++ j)
			fin_w1 >> weight[0][i][j];
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
