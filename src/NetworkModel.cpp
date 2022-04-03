#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>
#include <sstream>

#include "NetworkModel.h"

/*
std::vector<std::string> stringSplit(const std::string &str, const std::string &delim) {
	std::regex reg(delim);
	std::vector<std::string> sub_string(std::sregex_token_iterator(str.begin(), str.end(), reg, -1), 
										std::sregex_token_iterator());
	return sub_string;
}*/

Neuron::Neuron(int neuron_id, std::vector<int> output_neuron, std::vector<float> weight): 
	neuron_id(neuron_id), update_pointer(0) {
	for (int i = 0; i < output_neuron.size(); ++ i) {
		output_synapse[output_neuron[i]] = weight[i];
	}
}

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
	if (num_dim <= 3 && dim[0] <= 256 && num_dim >= 2 && dim[1] <= 64 && num_dim >= 3 && dim[2] <= 16 && dim[0] != 4) {
		large_scale = false;
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
		large_scale = true;
		std::ifstream fin_w(dir + "weights.txt");
		neuron_list.reserve(total_neuron);
		std::string read_buffer;
		// read from weight file
		for (int i = 0; i < total_neuron; ++ i) {
			getline(fin_w, read_buffer);
			std::istringstream is(read_buffer);
			std::vector <int> output_neuron;
			std::vector <float> weight;

			int neuron_id, neuron;
			float w;
			is >> neuron_id;

			while (is >> neuron) {
				is >> w;
				output_neuron.push_back(neuron);
				weight.push_back(w);
			}
			
			neuron_list.emplace_back(Neuron(neuron_id, output_neuron, weight));
		}

		//update input neuron
		for (int i = 0; i < total_neuron; ++ i) {
			int neuron_in = neuron_list[i].getNeuronId(); // should be i
			std::vector<int> output_neuron = neuron_list[i].getOutput();

			for (auto neuron_out: output_neuron)
				neuron_list[neuron_out].addInput(neuron_in);
		}
	}
}

int Neuron::getNeuronId() {
	return neuron_id;
}

void Neuron::addInput(int i) {
	input_neuron.push(i);
}

int Neuron::extractInput() {
	int ret = input_neuron.front();
	input_neuron.pop();
	return ret;
}

void Neuron::updateOutput(int src, int dst) {
	float w = output_synapse[src];
	output_synapse.erase(src);
	output_synapse[dst] = w;
}

int Neuron::getInputSize() {
	return input_neuron.size();
}

void NetworkModel::networkUnrolling(int num_connection) {
	// update all the neurons
	int iter_size = neuron_list.size();
	for (int i = 0; i < iter_size; ++ i) {
		// skip suitable neurons
		//if (i % 10 == 0)
		std::cout << std::endl << i << " of " << iter_size << " ";
		while (neuron_list[i].getInputSize() > num_connection) {
			// get inputs
			std::vector<int> grouped_input;
			for (int j = 0; j < num_connection; ++ j)
				grouped_input.push_back(neuron_list[i].extractInput());

			int neuron_id = neuron_list.size();
			std::vector<int> output_neuron = {neuron_list[i].getNeuronId()};
			std::vector<float> weight = {10.0f};

			//neuron_list.reserve(neuron_id + 1);
			neuron_list.emplace_back(Neuron(neuron_id, output_neuron, weight));
			neuron_list[i].addInput(neuron_id);

			//time_t begin_update, end_update;
			//begin_update = clock();
			for (auto input: grouped_input) {
				neuron_list[input].updateOutput(i, neuron_id);
				neuron_list[neuron_id].addInput(input);
			}

			std::cout << neuron_list[i].getInputSize() << " ";
			//end_update = clock();
			//double ret_time = (double)(end_update - begin_update) / CLOCKS_PER_SEC;
			//std::cout << "time: " << ret_time << std::endl;
		}
	}
}

std::vector<int> Neuron::getOutput() {
	std::vector <int> output_neuron;
	std::map<int, float>::iterator iter;
	for (iter = output_synapse.begin(); iter != output_synapse.end(); iter++)
		output_neuron.push_back(iter->first);
	return output_neuron;
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

std::vector<Neuron> NetworkModel::getNeuronList() {
	return neuron_list;
}