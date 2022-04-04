#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>
#include <sstream>
#include <random>

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
	input_settled = false;
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


std::vector<int>& Neuron::getInput() {
	if (!input_neuron_list.size())
		for (int i = 0; i < input_neuron.size(); ++ i) {
			input_neuron_list.push_back(input_neuron.front());
			input_neuron.pop();
			input_neuron.push(input_neuron_list[i]);
		}
	return input_neuron_list;
}

void NetworkModeling::clusteringUpdateSet(std::set<int> &input_set, std::set<int> &output_set) {
	std::set<int>::iterator set_it;
	
	for (set_it = input_set.begin(); set_it != input_set.end(); ++ set_it) {
		int input_id = *set_it;
		bool add_set = true;
		std::vector<int> output_list = neuron_list[input_id].getOutput();
		for (auto output_id: output_list) {
			std::vector<int> input_list = neuron_list[output_id].getInput();
			for (auto &input_id_of_output: input_list)
				if (!input_set.count(input_id_of_output)) {
					add_set = false;
					break;
				}
			// add if all input in the set
			if (add_set)
				output_set.insert(output_id);
		}	
	}
}

std::vector<pair<int, int>> NetworkModeling::getCluster(std::vector<std::set<int> >& neuron_set_list, std::vector<int>& dim) {
	// get cluster
	std::vector <pair <int, int> > neuron_cluster;
	int max_input_settled = 0;

	std::random_device rd;

	for (int T = 0; T < ITER_TIME; ++ T) {
		std::vector<std::set<int> > clustered_neuron;
		int input_settled_cnt = 0;
		// loop in neuron_sets
		for (int i = dim.size() - 1; i >= 0; -- i) {
			std::vector<int> neuron_in_this_layer;
			for (std::set<int>::iterator it = neuron_set_list[i].begin(); it != neuron_list[i].end(); ++ it)
				if (!clustered_neuron[i].count(*it))
					neuron_in_this_layer.push_back(*it);
			if (!neruon_in_this_layer.size())
				continue;
			std::uniform_int_distribution<int> ud(0, neuron_in_this_layer.size() - 1);
			std::vector<int> rest_dim(i + 1, 0);
			for (int t = 0; t < RAMDOM_TIME; ++ t) {
				int neuron_to_cluster = neuron_in_this_layer[ud(rd)];
				// compute needed dim
				std::queue<int> backforward_que;
				std::set<int> in_bf_que;
				backforward_que.push(neuron_to_cluster);
				in_bf_que.insert(neuron_to_cluster);
				int current_layer = i;
				while (!backforward_que.empty()) {
					// the front neuron is in this layer
					int front_neuron = backforward_que.front()
					if (neuron_set_list[current_layer].count(front_neuron)) {
						rest_dim[current_layer] ++;
						backforward_que.pop();
						
						//add all the related input into que
						std::vector<int> input_list = neuron_list[front_neuron].getInput();
						for (auto &input: input_list) {
							if (in_bf_que.count(input) || clustered_neuron[current_layer - 1].count(input))
								continue;
							backforward_que.push(input);
							in_bf_que.insert(input);
						}
					}
					// go to a new layer
					else
						current_layer --;
				}

				// break if found a valid neuron
				bool valid = true;
				for (int j = i; j >= 0; ++ j)
					if (clustered_neuron[i].size + rest_dim[i] > dim[i]) {
						valid = false;
						break;
					}
				if (valid) {
					t = 0;
					// BFS again to update cluster
					backforward_que.push(neuron_to_cluster);
					current_layer = i;
					while (!backforward_que.empty()) {
						int front_neuron = backforward_que.front()
						if (neuron_set_list[current_layer].count(front_neuron)) {
							backforward_que.pop();
							if (!clustered_neuron[current_layer].count(front_neuron)) {
								clustered_neuron[curret_layer].insert(front_neuron);
								input_settled_cnt ++;
							}
							
							std::vector<int> input_list = neuron_list[front_neuron].getInput();
							for (auto &input: input_list) {
								if (!clustered_neuron[current_layer - 1].count(input))
									backforward_que.push(input);
							}
						}
						else
							current_layer --;
					}
				}
			}
		}

		//update best
		if (input_settled_cnt > max_input_settled) {
			max_input_settled = input_settled_cnt;
			neuron_cluster.clear();
			for (int i = 0; i < dim.size(); ++ i) {
			}
		}
	}

	// update neuron_set_list
}

void NetworkModeling::ClusteringRemoveDummy(std::set<int> &input_set) {
	std::set<int>::iterator set_it = input_set.begin();

	while (set_it != input_set.end()) {
		int input_id = *set_it;
		bool dummy = true;
		std::vector<int> output_neuron = neuron_list[i].getOutput();
		for (auto &neuron: output_neuron)
			if (!neuron.input_settled) {
				dummy = false;
				break;
			}
		if (dummy)
			input_set.erase(input_id);
		else
			set_it ++;
	}
	
}

int NetworkModeling::networkClustering(std::vector<int> dim) {
	// divide neuron_list for target architecture
	std::vector<std::set<int> > neuron_set_list(dim.size());

	// initilize
	for (auto &neuron: neuron_list)
		if (!neuron.getInputSize()) {
			neuron_set_list[0].insert(neuron.getNeuronId());
			neuron.input_settled = true;
		}
	
	for (int i = 1; i < neuron_set_list.size(); i ++)
		clusteringUpdateSet(neuron_set_list[i - 1], neuron_set_list[i]);

	// loop until no input(has unknown children)
	while (neuron_set_list[0].size()) {
		neuron_cluster_list.emplace_back(getCluster(neuron_set_list, dim));
		ClusteringRemoveDummy(neuron_set_list[0]);
	}

	return neuron_cluster_list.size();
}

std::vector<int>& Neuron::getOutput() {
	if (!output_neuron.size()) {
		std::map<int, float>::iterator iter;
		for (iter = output_synapse.begin(); iter != output_synapse.end(); iter++)
			output_neuron.push_back(iter->first);
	}
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