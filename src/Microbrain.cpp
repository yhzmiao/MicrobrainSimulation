#include <carlsim.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cstring>
#include <ctime>

#include "Microbrain.h"
#include "NetworkModel.h"

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

		for (int i = 0; i < weight_pointer_list.size(); ++ i)
			for (int j = 0; j < weight_pointer_list[i].size(); ++ j)
				//for (int k = 0; k < 6; ++ k)
					delete [] weight_pointer_list[i][j];
		for (auto &wp: weight_pointer_i2l_ex) {
			delete [] wp[0];
			delete [] wp[1];
		}
		for (auto &wp: weight_pointer_i2l_in) {
			delete [] wp[0];
			delete [] wp[1];
		}
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
		ginput_all_ex.resize(1);
		ginput_all_in.resize(1);
		for (int i = 0; i < 1; ++ i) {
			//std::string ginput_all_name = "data_input_all_";
			ginput_all_ex[i] = sim.createSpikeGeneratorGroup("data_input_all_ex", grid_input_all, EXCITATORY_NEURON);
			ginput_all_in[i] = sim.createSpikeGeneratorGroup("data_input_all_in", grid_input_all, EXCITATORY_NEURON);
			sim.setSpikeMonitor(ginput_all_ex[i], "DEFAULT");
		}
		/*
		grid_single_neuron = Grid3D(1, 1, 1);
		ginput_single.resize(NUM_NEURON_LAYER1);
		glayer1_ex.resize(NUM_NEURON_LAYER1);
		glayer1_in.resize(NUM_NEURON_LAYER1);
		std::string input_name = "data_input_";
		std::string layer1_ex_name = "layer1_ex_";
		std::string layer1_in_name = "layer1_in_";

		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
			ginput_single[i] = sim.createSpikeGeneratorGroup(input_name + std::to_string(i), grid_single_neuron, EXCITATORY_NEURON);
			glayer1_ex[i] = sim.createGroupLIF(layer1_ex_name + std::to_string(i), grid_single_neuron, EXCITATORY_NEURON, 0, CPU_CORES);
			glayer1_in[i] = sim.createGroupLIF(layer1_in_name + std::to_string(i), grid_single_neuron, INHIBITORY_NEURON, 0, CPU_CORES);
			sim.setNeuronParametersLIF(glayer1_ex[i], 1.0f, 0.0f, 1.0f, 0.0f);
			sim.setNeuronParametersLIF(glayer1_in[i], 1.0f, 0.0f, 1.0f, 0.0f);
			sim.setSpikeMonitor(ginput_single[i], "DEFAULT");
			sim.setSpikeMonitor(glayer1_ex[i], "DEFAULT");
		}
		*/

		// layer1 ex and in
		
		grid_layer1_all_ex = Grid3D(NUM_NEURON_LAYER1, 1, 1);
		grid_layer1_all_in = Grid3D(NUM_NEURON_LAYER1, 1, 1);
		glayer1_all_ex = sim.createGroupLIF("layer1_all_ex", grid_layer1_all_ex, EXCITATORY_NEURON, 0, CPU_CORES);
		glayer1_all_in = sim.createGroupLIF("layer1_all_in", grid_layer1_all_in, INHIBITORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer1_all_ex, 1, 0, 1.0f, 0.0f);
		sim.setNeuronParametersLIF(glayer1_all_in, 1, 0, 1.0f, 0.0f);
		sim.setSpikeMonitor(glayer1_all_ex, "DEFAULT");
		sim.setSpikeMonitor(glayer1_all_in, "DEFAULT");
		

		// layer2 ex and in
		grid_layer2_all_ex = Grid3D(NUM_NEURON_LAYER2, 1, 1);
		grid_layer2_all_in = Grid3D(NUM_NEURON_LAYER2, 1, 1);
		glayer2_all_ex = sim.createGroupLIF("layer2_all_ex", grid_layer2_all_ex, EXCITATORY_NEURON, 0, CPU_CORES);
		glayer2_all_in = sim.createGroupLIF("layer2_all_in", grid_layer2_all_in, INHIBITORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer2_all_ex, 1, 0, 1.0f, 0.0f);
		sim.setNeuronParametersLIF(glayer2_all_in, 1, 0, 1.0f, 0.0f);
		sim.setSpikeMonitor(glayer2_all_ex, "DEFAULT");
		sim.setSpikeMonitor(glayer2_all_in, "DEFAULT");

		// layer3
		grid_layer3_all = Grid3D(NUM_NEURON_LAYER3, 1, 1);
		glayer3_all = sim.createGroupLIF("layer3_all", grid_layer3_all, EXCITATORY_NEURON, 0, CPU_CORES);
		sim.setNeuronParametersLIF(glayer3_all, 1, 0, 1.0f, 0.0f);
		result_monitor_layer1_ex = sim.setSpikeMonitor(glayer1_all_ex, "DEFAULT");
		result_monitor_layer1_in = sim.setSpikeMonitor(glayer1_all_in, "DEFAULT");
		result_monitor_layer2_ex = sim.setSpikeMonitor(glayer2_all_ex, "DEFAULT");
		result_monitor_layer2_in = sim.setSpikeMonitor(glayer2_all_in, "DEFAULT");
		result_monitor_layer3 = sim.setSpikeMonitor(glayer3_all, "DEFAULT");
	}
	
	std::cout << "Created Neurons!!" << std::endl;
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
		
		input_ex_to_layer1_ex_all.resize(1);
		input_in_to_layer1_ex_all.resize(1);
		input_ex_to_layer1_in_all.resize(1);
		input_in_to_layer1_in_all.resize(1);
		/*
		input_to_layer1_ex_all.resize(NUM_NEURON_LAYER1);
		input_to_layer1_in_all.resize(NUM_NEURON_LAYER1);
		for (int i = 0; i < 1; ++ i) {
			input_to_layer1_ex_all[i] = sim.connect(ginput_all[i], glayer1_all_ex, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
			input_to_layer1_in_all[i] = sim.connect(ginput_all[i], glayer1_all_in, "one-to-one", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		}
		*/
		input_ex_to_layer1_ex_all[0] = sim.connect(ginput_all_ex[0], glayer1_all_ex, "one-to-one", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		input_in_to_layer1_ex_all[0] = sim.connect(ginput_all_in[0], glayer1_all_ex, "one-to-one", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		input_ex_to_layer1_in_all[0] = sim.connect(ginput_all_ex[0], glayer1_all_in, "one-to-one", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		input_in_to_layer1_in_all[0] = sim.connect(ginput_all_in[0], glayer1_all_in, "one-to-one", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		/*
		input_to_layer1_ex.resize(NUM_NEURON_LAYER1);
		input_to_layer1_in.resize(NUM_NEURON_LAYER1);

		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
			input_to_layer1_ex[i] = sim.connect(ginput_single[i], glayer1_ex[i], "full", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
			input_to_layer1_in[i] = sim.connect(ginput_single[i], glayer1_in[i], "full", RangeWeight(10.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		}

		layer1_ex_to_layer2_ex.resize(NUM_NEURON_LAYER1);
		layer1_ex_to_layer2_in.resize(NUM_NEURON_LAYER1);
		layer1_in_to_layer2_ex.resize(NUM_NEURON_LAYER1);
		layer1_in_to_layer2_in.resize(NUM_NEURON_LAYER1);

		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
			layer1_ex_to_layer2_ex[i] = sim.connect(glayer1_ex[i], glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
			layer1_ex_to_layer2_in[i] = sim.connect(glayer1_ex[i], glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
			layer1_in_to_layer2_ex[i] = sim.connect(glayer1_in[i], glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
			layer1_in_to_layer2_in[i] = sim.connect(glayer1_in[i], glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC);
		}
		*/

		
		layer1_ex_to_layer2_ex_all.setConnectionValue(sim.connect(glayer1_all_ex, glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_ex_to_layer2_in_all.setConnectionValue(sim.connect(glayer1_all_ex, glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_in_to_layer2_ex_all.setConnectionValue(sim.connect(glayer1_all_in, glayer2_all_ex, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer1_in_to_layer2_in_all.setConnectionValue(sim.connect(glayer1_all_in, glayer2_all_in, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		

		layer2_ex_to_layer3_all.setConnectionValue(sim.connect(glayer2_all_ex, glayer3_all, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
		layer2_in_to_layer3_all.setConnectionValue(sim.connect(glayer2_all_in, glayer3_all, "full", RangeWeight(0.0f), 1.0f, RangeDelay(1), RadiusRF(-1), SYN_PLASTIC));
	}
}

void Microbrain::initInputWeightPointer(CARLsim &sim) {
	weight_pointer_i2l_ex.resize(NUM_NEURON_LAYER1);
	weight_pointer_i2l_in.resize(NUM_NEURON_LAYER1);
	std::pair<float *, int> tmp_pair;
	
	// initilize weight pointer for 0
	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
		weight_pointer_i2l_ex[i].resize(2);
		weight_pointer_i2l_in[i].resize(2);

		tmp_pair = sim.getWeightData(input_to_layer1_ex_all[i]);
		weight_pointer_i2l_ex[i][0] = new float[tmp_pair.second];
		memcpy(weight_pointer_i2l_ex[i][0], tmp_pair.first, sizeof(float) * tmp_pair.second);
		
		tmp_pair = sim.getWeightData(input_to_layer1_in_all[i]);
		weight_pointer_i2l_in[i][0] = new float[tmp_pair.second];
		memcpy(weight_pointer_i2l_in[i][0], tmp_pair.first, sizeof(float) * tmp_pair.second);
	}

	std::cout << "Finished Initilization 0!!" << std::endl;

	// initilize weight pointer for 1
	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
		sim.setWeight(input_to_layer1_ex_all[i], i, i, 5.0f, true);
		sim.setWeight(input_to_layer1_in_all[i], i, i, 5.0f, true);

		tmp_pair = sim.getWeightData(input_to_layer1_ex_all[i]);
		weight_pointer_i2l_ex[i][1] = new float[tmp_pair.second];
		memcpy(weight_pointer_i2l_ex[i][1], tmp_pair.first, sizeof(float) * tmp_pair.second);

		tmp_pair = sim.getWeightData(input_to_layer1_in_all[i]);
		weight_pointer_i2l_in[i][1] = new float[tmp_pair.second];
		memcpy(weight_pointer_i2l_in[i][1], tmp_pair.first, sizeof(float) * tmp_pair.second);

		input_size = tmp_pair.second;
	}

	std::cout << "Finished Initilization 1!!" << std::endl;

	// recover weight
	std::vector<float> input_matrix(NUM_NEURON_LAYER1, 1.0f);
	recoverInput(sim, input_matrix);
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

double Microbrain::loadWeight(CARLsim &sim, std::vector <std::vector <std::vector <float> > > &weight) {
	unsigned long int dim[3] = {weight[0].size(), weight[0][0].size(), 0};
	if (weight[1].size())
		dim[2] = weight[1][0].size();
	std::cout << dim[0] << " " << dim[1] << " " << dim[2] << std::endl;
	time_t begin_load, end_load;
	begin_load = clock();
	if (dim[0] && dim[1])
		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
			for (int j = 0; j < NUM_NEURON_LAYER2; ++ j) {
				
				float syn_weight = 0.0f;
				if (i < dim[0] && j < dim[1]) {
					//std::cout << i << " " << j << " " << weight[0][i][j] << std::endl;
					syn_weight = weight[0][i][j];
				}
				//if (dim[1] == 10)
				//std::cout << syn_weight << " ";
				
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
				
				/*
				if (syn_weight > -EPS) {
					sim.setWeight(layer1_ex_to_layer2_ex[i], 0, j, syn_weight, true);
					sim.setWeight(layer1_ex_to_layer2_in[i], 0, j, syn_weight, true);
					sim.setWeight(layer1_in_to_layer2_ex[i], 0, j, 0.0f, true);
					sim.setWeight(layer1_in_to_layer2_in[i], 0, j, 0.0f, true);
				}
				else {
					sim.setWeight(layer1_ex_to_layer2_ex[i], 0, j, 0.0f, true);
					sim.setWeight(layer1_ex_to_layer2_in[i], 0, j, 0.0f, true);
					sim.setWeight(layer1_in_to_layer2_ex[i], 0, j, -syn_weight, true);
					sim.setWeight(layer1_in_to_layer2_in[i], 0, j, -syn_weight, true);
				}
				*/
			}
			//if (dim[1] == 10)
			//std::cout << std::endl;
		}
	//std::cout << "Finished first part!" << std::endl;
	if (dim[1] && dim[2])
		for (int i = 0; i < NUM_NEURON_LAYER2; ++ i)
			for (int j = 0; j < NUM_NEURON_LAYER3; ++ j) {
				float syn_weight = 0.0f;
				if (i < dim[1] && j < dim[2])
					syn_weight = weight[1][i][j];
				if (syn_weight > -EPS) {
					sim.setWeight(layer2_ex_to_layer3_all.connection, i, j, syn_weight, true);
					sim.setWeight(layer2_in_to_layer3_all.connection, i, j, 0.0f, true);
				}
				else {
					sim.setWeight(layer2_ex_to_layer3_all.connection, i, j, 0.0f, true);
					sim.setWeight(layer2_in_to_layer3_all.connection, i, j, -syn_weight, true);
				}
			}
	end_load = clock();
	double ret_time = (double)(end_load - begin_load) / CLOCKS_PER_SEC;
	std::cout << "Loading time: " << ret_time << std::endl;
	return ret_time;
}

double Microbrain::loadWeight(CARLsim &sim, NetworkModel &network_model, int model_id, int cluster_id, bool &in_map) {
	//std::cout << "Start Loading!" << std::endl;
	time_t begin_load, end_load;
	double ret_time;
	begin_load = clock();
	
	if (weight_pointer_list.size() <= model_id)
		weight_pointer_list.resize(model_id + 1);

	// already load in map
	//std::cout << weight_pointer_list[model_id].size() << " " << cluster_id << std::endl;
	if (weight_pointer_list[model_id].size() > cluster_id) {
		in_map = true;
		float *tmp_pointer;
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id], sizeof(float) * weight_size);
		sim.replaceWeight(input_ex_to_layer1_ex_all[0], tmp_pointer);

		/* if not work continue this
		for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
			tmp_pointer = new float[weight_size_12];
			memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][i << 2], sizeof(float) * weight_size_12);
			sim.replaceWeight(input_to_layer1_ex[i], tmp_pointer);

			tmp_pointer = new float[weight_size_12];
			memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][i << 2 | 1], sizeof(float) * weight_size_12);
			sim.replaceWeight(input_to_layer1_in[i], tmp_pointer);
		}
		tmp_pointer = new float[weight_size_23];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][NUM_NEURON_LAYER1 << 2], sizeof(float) * weight_size_23);
		sim.replaceWeight(layer2_ex_to_layer3_all.connection, tmp_pointer);

		tmp_pointer = new float[weight_size_23];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][NUM_NEURON_LAYER1 << 2 | 1], sizeof(float) * weight_size_23);
		sim.replaceWeight(input_to_layer1_ex[NUM_NEURON_LAYER1 << 1 | 1], tmp_pointer);
		*/
		
		/*
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][0], sizeof(float) * weight_size);
		sim.replaceWeight(layer1_ex_to_layer2_ex_all.connection, tmp_pointer);
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][1], sizeof(float) * weight_size);
		sim.replaceWeight(layer1_ex_to_layer2_in_all.connection, tmp_pointer);
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][2], sizeof(float) * weight_size);
		sim.replaceWeight(layer1_in_to_layer2_ex_all.connection, tmp_pointer);
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][3], sizeof(float) * weight_size);
		sim.replaceWeight(layer1_in_to_layer2_in_all.connection, tmp_pointer);

		
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][4], sizeof(float) * weight_size);
		sim.replaceWeight(layer2_ex_to_layer3_all.connection, tmp_pointer);
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_list[model_id][cluster_id][5], sizeof(float) * weight_size);
		sim.replaceWeight(layer2_ex_to_layer3_all.connection, tmp_pointer);
		*/

		end_load = clock();
		ret_time = (double)(end_load - begin_load) / CLOCKS_PER_SEC;
		std::cout << "Loading time: " << ret_time << std::endl;
	}
	// not in map
	else {
		in_map = false;
		std::vector <std::vector <std::vector<float> > > weight;
		// setup weight metrix for microbrain
		network_model.setClusterWeight(cluster_id, weight);
		//std::cout << "Weight settled!" << std::endl;
		ret_time = loadWeight(sim, weight);
	}
	return ret_time;
}

void Microbrain::saveWeightPointer(CARLsim &sim, int model_id) {
	std::pair <float *, int> pointer_pair;
	float *weight_pointer;

	pointer_pair = sim.getWeightData(input_ex_to_layer1_ex_all[0]);
	weight_size = pointer_pair.second;
	weight_pointer = new float[weight_size];
	memcpy(weight_pointer, pointer_pair.first, sizeof(float) * weight_size);
	
	/*
	std::vector <float *> weight_pointer(6, nullptr);
	pointer_pair = sim.getWeightData(layer1_ex_to_layer2_ex_all.connection);
	weight_pointer[0] = new float[pointer_pair.second];
	memcpy(weight_pointer[0], pointer_pair.first, sizeof(float) * pointer_pair.second);
	pointer_pair = sim.getWeightData(layer1_ex_to_layer2_in_all.connection);
	weight_pointer[1] = new float[pointer_pair.second];
	memcpy(weight_pointer[1], pointer_pair.first, sizeof(float) * pointer_pair.second);
	pointer_pair = sim.getWeightData(layer1_in_to_layer2_ex_all.connection);
	weight_pointer[2] = new float[pointer_pair.second];
	memcpy(weight_pointer[2], pointer_pair.first, sizeof(float) * pointer_pair.second);
	pointer_pair = sim.getWeightData(layer1_in_to_layer2_in_all.connection);
	weight_pointer[3] = new float[pointer_pair.second];
	memcpy(weight_pointer[3], pointer_pair.first, sizeof(float) * pointer_pair.second);


	pointer_pair = sim.getWeightData(layer2_ex_to_layer3_all.connection);
	weight_pointer[4] = new float[pointer_pair.second];
	memcpy(weight_pointer[4], pointer_pair.first, sizeof(float) * pointer_pair.second);
	pointer_pair = sim.getWeightData(layer2_ex_to_layer3_all.connection);
	weight_pointer[5] = new float[pointer_pair.second];
	memcpy(weight_pointer[5], pointer_pair.first, sizeof(float) * pointer_pair.second);

	weight_size = pointer_pair.second;
	*/
	
	weight_pointer_list[model_id].emplace_back(weight_pointer);
	std::cout << "Saved Weight!!" << std::endl;
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
			sim.setWeight(input_to_layer1_ex_all[0], i, i, conn_weight, true);
			sim.setWeight(input_to_layer1_in_all[0], i, i, conn_weight, true);
		}
	}
	fin.close();
	if (!single_neuron_group) {
		result_monitor_layer3->startRecording();
		in.setRates(20000.0f / input_cnt);
		sim.setSpikeRate(ginput_all[0], &in);
	}
	//std::cout << input_cnt << std::endl;
}

float Microbrain::getWeightFromSpike(int num_spike, int run_time) {
	//if (num_spike)
	/*
	if(num_spike > 95)
		num_spike = 100;
	if (num_spike < 5)
		num_spike = 0;
	if (num_spike > 45 && num_spike < 55)
		num_spike = 50;
	*/

	return 1000.0f / run_time * num_spike;
	//return 0.01f;

	if (num_spike >= 75)
		return 10.0f;
	else if (num_spike >= 42)
		return 2.0f;
	else if (num_spike >= 29)
		return 1.1f;
	else if (num_spike >= 23)
		return 1.02f;
	else if (num_spike >= 18)
		return 1.005f;
	else if (num_spike >= 14)
		return 1.001f;
	else if (num_spike >= 11)
		return 1.0001f;
	else if (num_spike >= 9)
		return 1.00001f;
	else if (num_spike >= 5)
		return 1.000001f;

	return 0.0f;
}

void Microbrain::loadInput(CARLsim &sim, std::vector <std::pair<int, int> > &input_matrix) {
	int dim = input_matrix.size();
	float input_cnt = 0.0f;
	float *tmp_pointer;
	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
		if (i < dim) {
			//std::pair<float, float> weight = std::make_pair(getWeightFromSpike(input_matrix[i].first), getWeightFromSpike(input_matrix[i].second));

			sim.setWeight(input_ex_to_layer1_ex_all[0], i, i, 10.0f, true);
			sim.setWeight(input_ex_to_layer1_in_all[0], i, i, 10.0f, true);
			sim.setWeight(input_in_to_layer1_ex_all[0], i, i, 0.0f, true);
			sim.setWeight(input_in_to_layer1_in_all[0], i, i, 0.0f, true);
		}
		else {
			sim.setWeight(input_ex_to_layer1_ex_all[0], i, i, 0.0f, true);
			sim.setWeight(input_ex_to_layer1_in_all[0], i, i, 0.0f, true);
			sim.setWeight(input_in_to_layer1_ex_all[0], i, i, 0.0f, true);
			sim.setWeight(input_in_to_layer1_in_all[0], i, i, 0.0f, true);
		}
		/*
		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_i2l_ex[i][1], sizeof(float) * weight_size);
		sim.replaceWeight(input_to_layer1_ex_all[i], tmp_pointer);

		tmp_pointer = new float[weight_size];
		memcpy(tmp_pointer, weight_pointer_i2l_in[i][1], sizeof(float) * weight_size);
		sim.replaceWeight(input_to_layer1_in_all[i], tmp_pointer);
		*/
	}
}

void Microbrain::recoverInput(CARLsim &sim, std::vector <float> &input_matrix) {
	float *tmp_pointer;

	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) 
	 	if (input_matrix[i] > 0.5f) {
			 
			sim.setWeight(input_ex_to_layer1_ex_all[0], i, i, 0.0f, true);
			sim.setWeight(input_ex_to_layer1_in_all[0], i, i, 0.0f, true);
			sim.setWeight(input_in_to_layer1_ex_all[0], i, i, 0.0f, true);
			sim.setWeight(input_in_to_layer1_in_all[0], i, i, 0.0f, true);
			/*
			tmp_pointer = new float[input_size];
			memcpy(tmp_pointer, weight_pointer_i2l_ex[i][0], sizeof(float) * input_size);
			sim.replaceWeight(input_to_layer1_ex_all[i], tmp_pointer);

			tmp_pointer = new float[input_size];
			memcpy(tmp_pointer, weight_pointer_i2l_in[i][0], sizeof(float) * input_size);
			sim.replaceWeight(input_to_layer1_in_all[i], tmp_pointer);
			*/
		}

	//std::cout << input_size << std::endl;
}

std::vector<int> Microbrain::testResult(CARLsim &sim, std::vector<std::pair<int, int> > input_rate, PoissonRate &in, int run_time, float input_cnt) {
	result_monitor_layer1_ex->startRecording();
	result_monitor_layer1_in->startRecording();
	result_monitor_layer2_ex->startRecording();
	result_monitor_layer2_in->startRecording();
	result_monitor_layer3->startRecording();
	//in.setRates(40000.0f / input_cnt);
	//in.setRates(1000.0f);
	//input_rate[255].second = 100;
	std::vector <float> rates(NUM_NEURON_LAYER1, 0.0f);
	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
		rates[i] = getWeightFromSpike(input_rate[i].first, run_time);
	}
	in.setRates(rates);
	sim.setSpikeRate(ginput_all_ex[0], &in);
	sim.runNetwork(run_time / 1000, run_time % 1000);
	
	in.setRates(0.0f);
	sim.setSpikeRate(ginput_all_ex[0], &in);
	sim.runNetwork(0, 5);
	
	result_monitor_layer1_ex->stopRecording();
	result_monitor_layer1_in->stopRecording();
	result_monitor_layer2_ex->stopRecording();
	result_monitor_layer2_in->stopRecording();
	result_monitor_layer3->stopRecording();

	std::vector < std::vector <int> > result_vector_1_ex = result_monitor_layer1_ex->getSpikeVector2D();
	std::vector < std::vector <int> > result_vector_1_in = result_monitor_layer1_in->getSpikeVector2D();
	std::vector < std::vector <int> > result_vector_2_ex = result_monitor_layer2_ex->getSpikeVector2D();
	std::vector < std::vector <int> > result_vector_2_in = result_monitor_layer2_in->getSpikeVector2D();
	std::vector < std::vector <int> > result_vector_3 = result_monitor_layer3->getSpikeVector2D();

	std::vector<int> spike_time; 
	for (auto &result: result_vector_2_ex) {
		std::cout << result.size() << " ";
		spike_time.push_back(result.size());
	}
	std::cout << std::endl;
	for (auto &result: result_vector_2_in) {
		std::cout << result.size() << " ";
		spike_time.push_back(result.size());
	}
	std::cout << std::endl;
	for (auto &result: result_vector_3)
		spike_time.push_back(result.size());
	
	int total_spike_transformed = 0;
	for (auto &result: result_vector_1_ex)
		total_spike_transformed += result.size();
	for (auto &result: result_vector_1_in)
		total_spike_transformed += result.size();

	spike_time.push_back(total_spike_transformed);
	/*
	int max_num_spike = 0, max_spike_id;
	std::cout << "Output Spike";
	for (int i = 0; i < result_vector.size(); ++ i)
		std::cout << "\t" << i;
	std::cout << std::endl << "Num of Spike";
	for (int i = 0; i < result_vector.size(); ++ i) {
		std::cout << "\t" << result_vector[i].size();
		if (result_vector[i].size() > max_num_spike) {
			max_num_spike = result_vector[i].size();
			max_spike_id = i;
		}
	}
	*/
	//for (auto &st: spike_time)
	//	if (st <= 2)
	//		st = 0;
	return spike_time;
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
			sim.setWeight(input_to_layer1_ex_all[0], i, i, conn_weight, true);
			sim.setWeight(input_to_layer1_in_all[0], i, i, conn_weight, true);
		}
		
		result_monitor_layer3->startRecording();
		in.setRates(40000.0f / input_cnt);
		sim.setSpikeRate(ginput_all[0], &in);
		sim.runNetwork(0, 100);
		result_monitor_layer3->stopRecording();

		std::vector < std::vector <int> > result_vector = result_monitor_layer3->getSpikeVector2D();
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
	result_monitor_layer3->stopRecording();
	std::vector < std::vector <int> > result_vector = result_monitor_layer3->getSpikeVector2D();
	if (print_result) {
		//int max_num_spike = 0;
		for (int i = 0; i < result_vector.size(); ++ i) 
			std::cout << i << " " << result_vector[i].size() << std::endl;
		//std::cout << result_monitor->getNeuronNumSpikes(6) << std::endl;
	}
	return result_vector;
}

