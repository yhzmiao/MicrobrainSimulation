#include <carlsim.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "Microbrain.h"

Microbrain::Microbrain(bool exist_negative, bool recurrent, bool single_neuron_group) {
	this->exist_negative = exist_negative;
	this->recurrent = recurrent;
	this->single_neuron_group = single_neuron_group;

	if (single_neuron_group) {
		grid_input = new Grid3D[NUM_NEURON_LAYER1 << exist_negative];
		grid_layer1 = new Grid3D[NUM_NEURON_LAYER1 << exist_negative];
		grid_layer2 = new Grid3D[NUM_NEURON_LAYER2 << exist_negative];
		grid_layer3 = new Grid3D[NUM_NEURON_LAYER3 << exist_negative];

		ginput = new int[NUM_NEURON_LAYER1 << exist_negative];
		glayer1 = new int[NUM_NEURON_LAYER1 << exist_negative];
		glayer2 = new int[NUM_NEURON_LAYER2 << exist_negative];
		glayer3 = new int[NUM_NEURON_LAYER3 << exist_negative];

		input_to_layer1 = new Synapse[NUM_NEURON_LAYER1 << exist_negative];
		layer1_to_layer2 = new Synapse *[NUM_NEURON_LAYER1 << exist_negative];
		for (int i = 0; i < (NUM_NEURON_LAYER1 << exist_negative); ++ i)
			layer1_to_layer2[i] = new Synapse[NUM_NEURON_LAYER2 << exist_negative];
		
		// todo: add recurrent support
		// if (recurrent) {
		// }
		
		layer2_to_layer3 = new Synapse *[NUM_NEURON_LAYER2 << exist_negative];
		for (int i = 0; i < (NUM_NEURON_LAYER2 << exist_negative); ++ i)
			layer2_to_layer3[i] = new Synapse[NUM_NEURON_LAYER3 << exist_negative];
	}
}

Microbrain::~Microbrain() {
	if (single_neuron_group) {
		delete [] grid_input;
		delete [] grid_layer1;
		delete [] grid_layer2;
		delete [] grid_layer3;

		delete [] ginput;
		delete [] glayer1;
		delete [] glayer2;
		delete [] glayer3;

		delete [] input_to_layer1;
		for (int i = 0; i < (NUM_NEURON_LAYER1 << exist_negative); ++ i)
			delete [] layer1_to_layer2[i];
		delete [] layer1_to_layer2;

		// todo: add recurrent support

		for (int i = 0; i < (NUM_NEURON_LAYER2 << exist_negative); ++ i)
			delete [] layer2_to_layer3[i];
		delete [] layer2_to_layer3;
	}
}

void Microbrain::setupNeurons(CARLsim &sim) {
	if (single_neuron_group) {
		// input
		std::string ginput_name = "data_input_";
		for (int i = 0; i < NUM_NEURON_LAYER1; ++i) { // create and setup 256 one-neuron group
			grid_input[i] = Grid3D(1, 1, 1);
			ginput[i] = sim.createSpikeGeneratorGroup(ginput_name + std::to_string(i), grid_input[i], EXCITATORY_NEURON);
			sim.setSpikeMonitor(ginput[i], "DEFAULT");
		}

		//layer1
		std::string glayer1_name = "layer1_";
		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) { // create and setup 256 one-neuron group
			grid_layer1[i] = Grid3D(1, 1, 1);
			glayer1[i] = sim.createGroupLIF(glayer1_name + std::to_string(i), grid_layer1[i], EXCITATORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer1[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer1[i], "DEFAULT");
		}
		for (int i = NUM_NEURON_LAYER1; i < (NUM_NEURON_LAYER1 << exist_negative); ++ i) { // create and setup 256 one-neuron group
			grid_layer1[i] = Grid3D(1, 1, 1);
			glayer1[i] = sim.createGroupLIF(glayer1_name + std::to_string(i), grid_layer1[i], INHIBITORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer1[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer1[i], "DEFAULT");
		}

		//layer2
		std::string glayer2_name = "layer2_";
		for (int i = 0; i < NUM_NEURON_LAYER2; ++ i) { // create and setup 64 one-neuron group
			grid_layer2[i] = Grid3D(1, 1, 1);
			glayer2[i] = sim.createGroupLIF(glayer2_name + std::to_string(i), grid_layer2[i], EXCITATORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer2[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer2[i], "DEFAULT");
		}
		for (int i = NUM_NEURON_LAYER2; i < (NUM_NEURON_LAYER2 << exist_negative); ++ i) { // create and setup 64 one-neuron group
			grid_layer2[i] = Grid3D(1, 1, 1);
			glayer2[i] = sim.createGroupLIF(glayer2_name + std::to_string(i), grid_layer2[i], INHIBITORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer2[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer2[i], "DEFAULT");
		}

		//layer3
		std::string glayer3_name = "layer3_";
		for (int i = 0; i < NUM_NEURON_LAYER3; ++ i) { // create and setup 16 one-neuron group
			grid_layer3[i] = Grid3D(1, 1, 1);
			glayer3[i] = sim.createGroupLIF(glayer3_name + std::to_string(i), grid_layer3[i], EXCITATORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer3[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer3[i], "DEFAULT");
		}
		for (int i = NUM_NEURON_LAYER3; i < (NUM_NEURON_LAYER3 << exist_negative); ++ i) { // create and setup 16 one-neuron group
			grid_layer3[i] = Grid3D(1, 1, 1);
			glayer3[i] = sim.createGroupLIF(glayer3_name + std::to_string(i), grid_layer3[i], INHIBITORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer3[i], 20.0f, 5.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(glayer3[i], "DEFAULT");
		}
	}
	else {
		// input layer
		grid_input_all = Grid3D(NUM_NEURON_LAYER1, 1, 1);
		ginput_all = sim.createSpikeGeneratorGroup("data_input_all", grid_input_all, EXCITATORY_NEURON);
		sim.setSpikeMonitor(ginput_all, "DEFAULT");

		// layer1 ex and in
		grid_layer1_all_ex = Grid3D(NUM_NEURON_LAYER1, 1, 1);
		grid_layer1_all_in = Grid3D(NUM_NEURON_LAYER1, 1, 1);
		glayer1_all_ex = sim.createGroupLIF("layer1_all_ex", grid_layer1_all_ex, EXCITATORY_NEURON, 0, CPU_CORES);
		glayer1_all_in = sim.createGroupLIF("layer1_all_in", grid_layer1_all_in, INHIBITORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer1_all_ex, 1.0f, 0.0f, 1.0f, 0.0f);
		sim.setNeuronParametersLIF(glayer1_all_in, 1.0f, 0.0f, 1.0f, 0.0f);
		sim.setSpikeMonitor(glayer1_all_ex, "DEFAULT");
		sim.setSpikeMonitor(glayer1_all_in, "DEFAULT");

		// layer2 ex and in
		grid_layer2_all_ex = Grid3D(NUM_NEURON_LAYER2, 1, 1);
		grid_layer2_all_in = Grid3D(NUM_NEURON_LAYER2, 1, 1);
		glayer2_all_ex = sim.createGroupLIF("layer2_all_ex", grid_layer2_all_ex, EXCITATORY_NEURON, 0, CPU_CORES);
		glayer2_all_in = sim.createGroupLIF("layer2_all_in", grid_layer2_all_in, INHIBITORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer2_all_ex, 1.0f, 0.0f, 1.0f, 0.0f);
		sim.setNeuronParametersLIF(glayer2_all_in, 1.0f, 0.0f, 1.0f, 0.0f);
		sim.setSpikeMonitor(glayer2_all_ex, "DEFAULT");
		sim.setSpikeMonitor(glayer2_all_in, "DEFAULT");

		// layer3
		grid_layer3_all = Grid3D(NUM_NEURON_LAYER3, 1, 1);
		glayer3_all = sim.createGroupLIF("layer3_all", grid_layer3_all, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer3_all, 1.0f, 0.0f, 1.0f, 0.0f);
		result_monitor = sim.setSpikeMonitor(glayer3_all, "DEFAULT");
	}
}

void Microbrain::Synapse::setValue(int c, float w) {
	connection = c;
	weight = w;
}

float Microbrain::Synapse::setWeight(float w) {
	return weight = w;
}

void Microbrain::SynapseGroup::setConnectionValue(int c){
	connection = c;
}

float Microbrain::SynapseGroup::setWeight(int x, int y, float w) {
	return weight[x][y] = w;
}

void Microbrain::setupConnections(CARLsim &sim) {
	if (single_neuron_group) {
		// input to layer1
		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i)
			input_to_layer1[i].setValue(sim.connect(ginput[i], glayer1[i], "full", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC), 10.0f);
		for (int i = NUM_NEURON_LAYER1; i < (NUM_NEURON_LAYER1 << exist_negative); ++ i)
			input_to_layer1[i].setValue(sim.connect(ginput[i - NUM_NEURON_LAYER1], glayer1[i], "full", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC), 10.0f);

		//layer1 to layer2
		for (int i = 0; i < (NUM_NEURON_LAYER1 << exist_negative); ++ i)
			for (int j = 0; j < (NUM_NEURON_LAYER2 << exist_negative); ++ j)
				layer1_to_layer2[i][j].setValue(sim.connect(glayer1[i], glayer2[j], "full", RangeWeight(1.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC), 0.0f);

		// todo: add recurrent 

		// layer2 to layer3
		for (int i = 0; i < (NUM_NEURON_LAYER2 << exist_negative); ++ i)
			for (int j = 0; j < (NUM_NEURON_LAYER3 << exist_negative); ++ j)
				layer2_to_layer3[i][j].setValue(sim.connect(glayer2[i], glayer3[j], "full", RangeWeight(1.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC), 0.0f);
	}
	else {
		input_to_layer1_ex_all.setConnectionValue(sim.connect(ginput_all, glayer1_all_ex, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		input_to_layer1_in_all.setConnectionValue(sim.connect(ginput_all, glayer1_all_in, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));

		layer1_ex_to_layer2_ex_all.setConnectionValue(sim.connect(glayer1_all_ex, glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_ex_to_layer2_in_all.setConnectionValue(sim.connect(glayer1_all_ex, glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_in_to_layer2_ex_all.setConnectionValue(sim.connect(glayer1_all_in, glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_in_to_layer2_in_all.setConnectionValue(sim.connect(glayer1_all_in, glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));

		layer2_ex_to_layer3_all.setConnectionValue(sim.connect(glayer2_all_ex, glayer3_all, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer2_in_to_layer3_all.setConnectionValue(sim.connect(glayer2_all_in, glayer3_all, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
	}
}

void Microbrain::loadWeight(CARLsim &sim, std::string &model_name, std::vector<int> &dim) {
	std::string dir = "model/Models/" + model_name + "/";
	
	// todo: throw some error here
	if (dim[0] && dim[1]) {
		std::ifstream fin(dir + "weights1.txt");
		
		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i)
			for (int j = 0; j < NUM_NEURON_LAYER2; ++ j) {
				float syn_weight = 0.0f;
				if (i < dim[0] && j < dim[1])
					fin >> syn_weight;
				//syn_weight *= 1;
				if (single_neuron_group) {
					syn_weight *= 10;
					if (syn_weight > -EPS) {
						sim.setWeight(layer1_to_layer2[i][j].connection, 0, 0, layer1_to_layer2[i][j].setWeight(syn_weight), true); //setWeight(connectionid, id, id, update)
						if (exist_negative)
							sim.setWeight(layer1_to_layer2[i][j + NUM_NEURON_LAYER2].connection, 0, 0, layer1_to_layer2[i][j + NUM_NEURON_LAYER2].setWeight(syn_weight), true);
					}
					else {
						sim.setWeight(layer1_to_layer2[i + NUM_NEURON_LAYER1][j].connection, 0, 0, layer1_to_layer2[i + NUM_NEURON_LAYER1][j].setWeight(-syn_weight), true);
						sim.setWeight(layer1_to_layer2[i + NUM_NEURON_LAYER1][j + NUM_NEURON_LAYER2].connection, 0, 0, layer1_to_layer2[i + NUM_NEURON_LAYER1][j + NUM_NEURON_LAYER2].setWeight(-syn_weight), true);
					}
				}
				else {
					// a positive value -> from ex to ex and in
					if (syn_weight > -EPS) {
						sim.setWeight(layer1_ex_to_layer2_ex_all.connection, i, j, syn_weight, true);
						sim.setWeight(layer1_ex_to_layer2_in_all.connection, i, j, syn_weight, true);
						sim.setWeight(layer1_in_to_layer2_ex_all.connection, i, j, 0.0f, true);
						sim.setWeight(layer1_in_to_layer2_in_all.connection, i, j, 0.0f, true);
					}
					else {
						sim.setWeight(layer1_ex_to_layer2_ex_all.connection, i, j, 0.0f, true);
						sim.setWeight(layer1_ex_to_layer2_in_all.connection, i, j, 0.0f, true);
						sim.setWeight(layer1_in_to_layer2_ex_all.connection, i, j, -syn_weight, true);
						sim.setWeight(layer1_in_to_layer2_in_all.connection, i, j, -syn_weight, true);
					}
				}
				//std::cout << i << " " << j << " " << syn_weight << std::endl;
			}
		fin.close();
	}

	// todo: add recurrent

	if (dim[1] && dim[2]) {
		std::ifstream fin(dir + "weights2.txt");

		for (int i = 0; i < NUM_NEURON_LAYER2; ++ i)
			for (int j = 0; j < NUM_NEURON_LAYER3; ++ j) {
				float syn_weight = 0.0f;
				if (i < dim[1] && j < dim[2])
					fin >> syn_weight;
				//syn_weight *= 1;
				if (single_neuron_group) {
					if (syn_weight > -EPS) {
						sim.setWeight(layer2_to_layer3[i][j].connection, 0, 0, layer2_to_layer3[i][j].setWeight(syn_weight), true); //setWeight(connectionid, id, id, update)
						if (exist_negative)
							sim.setWeight(layer2_to_layer3[i][j + NUM_NEURON_LAYER3].connection, 0, 0, layer2_to_layer3[i][j + NUM_NEURON_LAYER3].setWeight(syn_weight), true);
					}
					else {
						sim.setWeight(layer2_to_layer3[i + NUM_NEURON_LAYER2][j].connection, 0, 0, layer2_to_layer3[i + NUM_NEURON_LAYER2][j].setWeight(-syn_weight), true);
						sim.setWeight(layer2_to_layer3[i + NUM_NEURON_LAYER2][j + NUM_NEURON_LAYER3].connection, 0, 0, layer2_to_layer3[i + NUM_NEURON_LAYER2][j + NUM_NEURON_LAYER3].setWeight(-syn_weight), true);
					}
				}
				else {
					if (syn_weight > -EPS) {
						sim.setWeight(layer2_ex_to_layer3_all.connection, i, j, syn_weight, true);
						sim.setWeight(layer2_in_to_layer3_all.connection, i, j, 0.0f, true);
					}
					else {
						sim.setWeight(layer2_ex_to_layer3_all.connection, i, j, 0.0f, true);
						sim.setWeight(layer2_in_to_layer3_all.connection, i, j, -syn_weight, true);
					}
				}
			}
		fin.close();
	}
}

// todo: add more function
void Microbrain::loadInput(CARLsim &sim, std::string &dataset_name, float *input_matrix, int dim, int index, PoissonRate &in) {
	float input_cnt = 0.0f;
	std::ifstream fin("model/Datasets/" + dataset_name + "/input.txt");
	for (int i = 0; i < dim * index; ++ i)
		fin >> input_matrix[0];
	for (int i = 0; i < dim; ++ i) {
		fin >> input_matrix[i];
		//std::cout << i << " " <<input_matrix[i] << std::endl;
		input_cnt += input_matrix[i];
		if (single_neuron_group) {
			if (input_matrix[i] > 0.5f)
				sim.setSpikeRate(ginput[i], &in);
		}
		else {
			float conn_weight = input_matrix[i] * 10.0f;
			sim.setWeight(input_to_layer1_ex_all.connection, i, i, conn_weight, true);
			sim.setWeight(input_to_layer1_in_all.connection, i, i, conn_weight, true);
		}
	}
	fin.close();
	if (!single_neuron_group) {
		result_monitor->startRecording();
		in.setRates(20000.0f / input_cnt);
		sim.setSpikeRate(ginput_all, &in);
	}
	//std::cout << input_cnt << std::endl;
}

float Microbrain::testAccuracy(CARLsim &sim, std::string &dataset_name, float *input_matrix, int dim, int num_case, PoissonRate &in) {
	float correct_case = 0;
	std::ifstream fin("model/Datasets/" + dataset_name + "/input.txt");
	std::ifstream fin_res("model/Datasets/" + dataset_name + "/output.txt");

	for (int T = 0; T < num_case; ++ T) {
		float input_cnt = 0.0f;
		for (int i = 0; i < dim; ++ i) {
			fin >> input_matrix[i];
			//std::cout << i << " " <<input_matrix[i] << std::endl;
			input_cnt += input_matrix[i];
			float conn_weight = input_matrix[i] * 10.0f;
			sim.setWeight(input_to_layer1_ex_all.connection, i, i, conn_weight, true);
			sim.setWeight(input_to_layer1_in_all.connection, i, i, conn_weight, true);
		}
		
		result_monitor->startRecording();
		in.setRates(40000.0f / input_cnt);
		sim.setSpikeRate(ginput_all, &in);
		sim.runNetwork(0, 100);
		result_monitor->stopRecording();

		std::vector < std::vector <int> > result_vector = result_monitor->getSpikeVector2D();
		int max_num_spike = 0, max_spike_id;
		for (int i = 0; i < result_vector.size(); ++ i)
			if (result_vector[i].size() > max_num_spike) {
				max_num_spike = result_vector[i].size();
				max_spike_id = i;
			}
		int output_result;
		fin_res >> output_result;
		correct_case += output_result == max_spike_id;
	}
	fin.close();
	fin_res.close();
	return correct_case / num_case;
}


std::vector < std::vector <int> > Microbrain::getResults(bool print_result) {
	result_monitor->stopRecording();
	std::vector < std::vector <int> > result_vector = result_monitor->getSpikeVector2D();
	if (print_result) {
		//int max_num_spike = 0;
		for (int i = 0; i < result_vector.size(); ++ i) 
			std::cout << i << " " << result_vector[i].size() << std::endl;
		//std::cout << result_monitor->getNeuronNumSpikes(6) << std::endl;
	}
	return result_vector;
}

