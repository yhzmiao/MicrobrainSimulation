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
#include <set>

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
	lazy_update = false;
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

		// layer0 dim[0]  layer1 dim[1]  layer2 dim[2]: create neuron_list
		neuron_list.reserve(total_neuron);
		int s = 0, t_l = 0, t_r = dim[0];
		for (int i = 0; i < num_dim - 1; ++ i) {
			std::vector<int> output_neuron;
			t_l = t_r; t_r += dim[i + 1];
			for (int j = t_l; j < t_r; ++ j) {
				output_neuron.push_back(j);
				//std::cout << j << " ";
			}
			//std::cout << output_neuron.size() << std::endl;
			//std::cout << std::endl;
			for (int j = s; j < t_l; ++ j) {
				//std::cout << s << " " << j - s << " " << i << " " << weight[i].size() << std::endl;
				neuron_list.emplace_back(Neuron(j, output_neuron, weight[i][j - s]));
			}
			s = t_l;
		}
		
		std::vector <int> output_neuron;
		std::vector <float> weight;
		for (int i = t_l; i < t_r; ++ i)
			neuron_list.emplace_back(Neuron(i, output_neuron, weight));
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

	}
	//std::cout << "Finished initialization" << std::endl;
	//update input neuron
	for (int i = 0; i < total_neuron; ++ i) {
		int neuron_in = neuron_list[i].getNeuronId(); // should be i
		std::vector<int> output_neuron = neuron_list[i].getOutput();
		
		for (auto neuron_out: output_neuron) {
			//std::cout << neuron_out << std::endl;
			neuron_list[neuron_out].addInput(neuron_in);
		}
	}
	/*
	std::cout << "Finished update" << std::endl;

	for (int i = 0; i < total_neuron; ++ i) {
		std::vector<int> output_neuron = neuron_list[i].getOutput();
		std::cout << i << " ";
		for (auto neuron_out: output_neuron) {
			std:: cout << neuron_out << " ";
		}
		std::cout << std::endl;
	}
	*/
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
	//std::cout << "Swapping " << src << " " << dst << std::endl;
	float w = output_synapse[src];
	output_synapse.erase(src);
	output_synapse[dst] = w;
	output_neuron.clear();
}

int Neuron::getInputSize() {
	return input_neuron.size();
}

void NetworkModel::networkUnrolling(int num_connection) {
	// skip the process if not large scale
	if (!large_scale)
		return;
	// update all the neurons
	int iter_size = neuron_list.size();
	for (int i = 0; i < iter_size; ++ i) {
		// skip suitable neurons
		//if (i % 10 == 0)
		//std::cout << std::endl << i << " of " << iter_size << " ";
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


			//std::cout << neuron_list[i].getInputSize() << " ";
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

void NetworkModel::clusteringUpdateSet(std::set<int> &input_set, std::set<int> &output_set) {
	std::set<int>::iterator set_it;
	
	for (set_it = input_set.begin(); set_it != input_set.end(); ++ set_it) {
		int input_id = *set_it;
		bool add_set = true;
		//std::cout << "Now working on " << input_id << std::endl;
		std::vector<int> output_list = neuron_list[input_id].getOutput();
		for (auto output_id: output_list) {
			if(neuron_list[output_id].input_settled || output_set.count(output_id))
				continue;
			//std::cout << output_id << " ";
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
		//std::cout << std::endl;
	}
}

std::vector<std::pair<int, int>> NetworkModel::getCluster(std::vector<std::set<int> >& neuron_set_list, std::vector<int>& dim) {
	// get cluster
	std::vector <std::pair <int, int> > neuron_cluster;
	int max_input_settled = 0;

	std::random_device rd;

	for (int T = 0; T < ITER_TIME; ++ T) {
		//std::cout << "=======================================" << std::endl;
		//std::cout << "T = " << T << std::endl;
		std::vector<std::set<int> > clustered_neuron(dim.size());
		int input_settled_cnt = 0;
		// loop in neuron_sets
		for (int i = dim.size() - 1; i > 0; -- i) {
			std::vector<int> neuron_in_this_layer;
			for (std::set<int>::iterator it = neuron_set_list[i].begin(); it != neuron_set_list[i].end(); ++ it)
				if (!clustered_neuron[i].count(*it))
					neuron_in_this_layer.push_back(*it);
			//std::cout << "Neuron in layer " << i << " is " << neuron_in_this_layer.size() << std::endl;
			if (!neuron_in_this_layer.size())
				continue;
			std::uniform_int_distribution<int> ud(0, neuron_in_this_layer.size() - 1);
			std::vector<int> rest_dim(i + 1, 0);
			for (int t = 0; t < RANDOM_TIME; ++ t) {
			//for (int t = 0; t < neuron_in_this_layer.size(); ++ t) {
				int neuron_to_cluster = neuron_in_this_layer[ud(rd)];
				//int neuron_to_cluster = neuron_in_this_layer[t];
				if (clustered_neuron[i].count(neuron_to_cluster))
					continue;
				//std::cout << neuron_to_cluster << std::endl;
				// compute needed dim
				std::queue<int> backforward_que;
				std::set<int> in_bf_que;
				backforward_que.push(neuron_to_cluster);
				in_bf_que.insert(neuron_to_cluster);
				int current_layer = i;
				while (!backforward_que.empty()) {
					// the front neuron is in this layer
					int front_neuron = backforward_que.front();
					//std::cout << front_neuron << std::endl;
					if (neuron_set_list[current_layer].count(front_neuron)) {
						//std::cout << "++" << front_neuron << std::endl;
						rest_dim[current_layer] ++;
						backforward_que.pop();
						//add all the related input into que
						if (!current_layer)
							continue;
						std::vector<int> input_list = neuron_list[front_neuron].getInput();
						for (auto &input: input_list) {
							if (in_bf_que.count(input) || clustered_neuron[current_layer - 1].count(input))
								continue;
							backforward_que.push(input);
							in_bf_que.insert(input);
							//std::cout << "**" << input << std::endl;
						}
					}
					// go to a new layer
					else
						current_layer --;
				}

				// break if found a valid neuron
				bool valid = true;
				for (int j = i; j >= 0; -- j) {
					//std::cout << "**" << i << " " << clustered_neuron[i].size() << " " << rest_dim[i] << " " << dim[i] << std::endl;
					if (clustered_neuron[j].size() + rest_dim[j] > dim[j]) {
						valid = false;
						break;
					}
				}
				//std::cout << clustered_neuron[0].size() << " " << rest_dim[0] << std::endl;
				if (valid) {
					t = 0;
					// std::cout << valid << std::endl;
					// BFS again to update cluster
					backforward_que.push(neuron_to_cluster);
					current_layer = i;
					while (!backforward_que.empty()) {
						int front_neuron = backforward_que.front();
						//std::cout << front_neuron << std::endl;
						if (neuron_set_list[current_layer].count(front_neuron)) {
							backforward_que.pop();
							rest_dim[current_layer] --;

							if (!clustered_neuron[current_layer].count(front_neuron)) {
								clustered_neuron[current_layer].insert(front_neuron);
								input_settled_cnt ++;
							}
							
							if (!current_layer)
								continue;
							std::vector<int> input_list = neuron_list[front_neuron].getInput();
							for (auto &input: input_list) {
								if (!clustered_neuron[current_layer - 1].count(input))
									backforward_que.push(input);
							}
						}
						else
							current_layer --;
					}

					//some optimization
					for (auto &neuron: neuron_in_this_layer)
						if (!clustered_neuron[i].count(neuron)) {
							std::vector<int> input_list = neuron_list[neuron].getInput();
							bool free_to_add = true;
							for (auto &input_neuron: input_list)
								if (!clustered_neuron[i - 1].count(input_neuron)){
									free_to_add = false;
									break;
								}
							if (free_to_add && clustered_neuron[i].size() < dim[i]) {
								clustered_neuron[i].insert(neuron);
								input_settled_cnt;
								neuron_list[neuron].lazy_update = true;
							}
						}
				}
			}
		}
		//std::cout << "Found a cluster!" << std::endl;

		//update best
		if (input_settled_cnt > max_input_settled) {
			max_input_settled = input_settled_cnt;
			neuron_cluster.clear();
			for (int i = 0; i < dim.size(); ++ i) {
				for (std::set<int>::iterator it = clustered_neuron[i].begin(); it != clustered_neuron[i].end(); ++ it) {
					neuron_cluster.emplace_back(std::make_pair(i, *it));
				}
			}
		}
	}
	// update neuron_set_list
	for (auto &neuron_pair: neuron_cluster)
		if (neuron_pair.first) {
			neuron_set_list[neuron_pair.first].erase(neuron_pair.second);
			neuron_set_list[0].insert(neuron_pair.second);
			neuron_list[neuron_pair.second].input_settled = true;
		}
		
	// print cluster found
	std::cout << "Cluster Found is:" << std::endl;
	for (auto &neuron_pair: neuron_cluster) {
		std::cout << neuron_pair.first << " " << neuron_pair.second << std::endl;
	}

	return neuron_cluster;
}

void NetworkModel::ClusteringRemoveDummy(std::set<int> &input_set) {
	std::set<int>::iterator set_it = input_set.begin();
	while (set_it != input_set.end()) {
		int input_id = *set_it;
		bool dummy = true;
		std::vector<int> output_neuron = neuron_list[input_id].getOutput();
		for (auto &neuron_id: output_neuron)
			if (!neuron_list[neuron_id].input_settled) {
				dummy = false;
				break;
			}
		//std::cout << input_id << std::endl;
		if (dummy)
			input_set.erase(set_it++);
		else
			++ set_it;
	}
	
}

int NetworkModel::networkClustering(std::vector<int> dim) {
	if (!large_scale) {
		int s = 0;
		neuron_cluster_list.resize(1);
		for (int i = 0; i < this->dim.size(); ++ i) {
			for (int j = s; j < s + this->dim[i]; ++ j) {
				neuron_cluster_list[0].emplace_back(std::make_pair(i, j));
			}
			s += dim[i];
		}

		//for (auto &cluster: neuron_cluster_list[0])
		//	std::cout << cluster.first << " " << cluster.second << std::endl;
		return 1;
	}
	// divide neuron_list for target architecture
	neuron_cluster_list.resize(0);
	std::vector<std::set<int> > neuron_set_list(dim.size());

	// initilize
	for (auto &neuron: neuron_list) {
		neuron.input_settled = false;
		if (!neuron.getInputSize()) {
			neuron_set_list[0].insert(neuron.getNeuronId());
			neuron.input_settled = true;
		}
	}
	
	for (int i = 1; i < neuron_set_list.size(); i ++)
		clusteringUpdateSet(neuron_set_list[i - 1], neuron_set_list[i]);

	std::cout << "Finished Initilization!!" << std::endl;
	// loop until no input(has unknown children)
	int cnt = 0;
	while (neuron_set_list[0].size()) {
		//std::cout << neuron_set_list[0].size() << " " << neuron_set_list[1].size() << std::endl;
		neuron_cluster_list.emplace_back(getCluster(neuron_set_list, dim));
		std::cout << "ID: " << cnt++ << std::endl;
		//std::cout << neuron_set_list[0].size() << std::endl;
		ClusteringRemoveDummy(neuron_set_list[0]);
		//std::cout << neuron_set_list[0].size() << std::endl;
		for (int i = 1; i < neuron_set_list.size(); i ++)
			clusteringUpdateSet(neuron_set_list[i - 1], neuron_set_list[i]);
	}

	return neuron_cluster_list.size();
}

std::vector<int>& Neuron::getOutput() {
	if (!output_neuron.size()) {
		//std::cout << "here" << std::endl;
		std::map<int, float>::iterator iter;
		//output_neuron.clear();
		for (iter = output_synapse.begin(); iter != output_synapse.end(); iter++) {
			//std::cout << iter->first << " ";
			output_neuron.push_back(iter->first);
		}
		//std::cout << std::endl;
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