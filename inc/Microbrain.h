#ifndef _MICROBRAIN_H
#define _MICROBRAIN_H

#include "NetworkModel.h"

#define NUM_NEURON_LAYER1 256
#define NUM_NEURON_LAYER2 64
#define NUM_NEURON_LAYER3 16

#define EPS 1e-8



class Microbrain {
	public:
		Microbrain(bool no_negative = true, bool recurrent = false, bool single_neuron_group = false);
		~Microbrain();

		void setupNeurons(CARLsim &sim);
		void setupConnections(CARLsim &sim);

		void loadWeight(CARLsim &sim, std::string &model_name, std::vector<int> &dim);
		double loadWeight(CARLsim &sim, std::vector <std::vector <std::vector <float> > > &weight);
		double loadWeight(CARLsim &sim, NetworkModel &network_model, int model_id, int cluster_id, bool in_map);

		void saveWeightPointer(CARLsim &sim, int model_id);

		void loadInput(CARLsim &sim, std::string &dataset_name, float *input_matrix,int dim, int index, PoissonRate &in);
		void loadInput(CARLsim &sim, std::vector <std::pair<int, int>> &input_matrix);
		void recoverInput(CARLsim &sim, std::vector <float> &input_matrix);

		std::vector < std::vector <int> > getResults(bool print_result = true);
		float testAccuracy(CARLsim &sim, std::string &dataset_name, float *input_matrix, int dim, int num_case, PoissonRate &in);
		std::vector<int> testResult(CARLsim &sim, std::vector<std::pair<int, int> > input_rate, PoissonRate &in, int run_time, float input_cnt = 0.0f);

		void initInputWeightPointer(CARLsim &sim);
		float getWeightFromSpike(int num_spike, int run_time);

		struct Synapse {
			int connection;
			float weight;
			void setValue(int c, float w);
			float setWeight(float w);
		};

		struct SynapseGroup {
			int connection;
			std::vector < std::vector <float> > weight;
			
			void setConnectionValue(int c);
			float setWeight(int x, int y, float w);
		};
	private:
		// neurons of mubrain
		Grid3D *grid_input;
		Grid3D *grid_layer1;
		Grid3D *grid_layer2;
		Grid3D *grid_layer3;
		int *ginput;
		int *glayer1;
		int *glayer2;
		int *glayer3;

		// connections of mubrain
		Synapse *input_to_layer1;
		Synapse **layer1_to_layer2;
		Synapse **layer1_to_layer1;
		Synapse **layer2_to_layer3;

		// manage neurons together
		Grid3D grid_input_all;
		Grid3D grid_layer1_all_ex;
		Grid3D grid_layer1_all_in;
		Grid3D grid_layer2_all_ex;
		Grid3D grid_layer2_all_in;
		Grid3D grid_layer3_all;
		
		std::vector<int> ginput_all;
		std::vector<int> ginput_all_ex;
		std::vector<int> ginput_all_in;
		int glayer1_all_ex;
		int glayer1_all_in;
		int glayer2_all_ex;
		int glayer2_all_in;
		int glayer3_all;

		// manage connections together
		// todo: maybe store the weight of connections
		//SynapseGroup input_to_layer1_ex_all;
		//SynapseGroup input_to_layer1_in_all;
		std::vector <int> input_to_layer1_ex_all;
		std::vector <int> input_to_layer1_in_all;
		
		// draft version
		std::vector <int> input_ex_to_layer1_ex_all;
		std::vector <int> input_in_to_layer1_ex_all;
		std::vector <int> input_ex_to_layer1_in_all;
		std::vector <int> input_in_to_layer1_in_all;

		//changed first layer into single neuron
		Grid3D grid_single_neuron;
		std::vector<int> ginput_single;

		std::vector<int> glayer1_ex;
		std::vector<int> glayer1_in;

		std::vector <int> input_to_layer1_ex;
		std::vector <int> input_to_layer1_in;
		
		std::vector <int> layer1_ex_to_layer2_ex;
		std::vector <int> layer1_ex_to_layer2_in;
		std::vector <int> layer1_in_to_layer2_ex;
		std::vector <int> layer1_in_to_layer2_in;

		// weight pointer for input layer
		std::vector<std::vector<float *> > weight_pointer_i2l_ex;
		std::vector<std::vector<float *> > weight_pointer_i2l_in;
		int input_size;

		SynapseGroup layer1_ex_to_layer2_ex_all;
		SynapseGroup layer1_ex_to_layer2_in_all;
		SynapseGroup layer1_in_to_layer2_ex_all;
		SynapseGroup layer1_in_to_layer2_in_all;
		// SynapseGroup layer1_to_layer1_all;
		SynapseGroup layer2_ex_to_layer3_all;
		SynapseGroup layer2_in_to_layer3_all;

		// model weight pointer
		std::vector<std::vector<float *> > weight_pointer_list;
		//int weight_size_12, weight_size_23;
		int weight_size;

		// configurations
		bool exist_negative;
		bool recurrent;
		bool single_neuron_group;

		// spike monitor
		SpikeMonitor * result_monitor_layer2_ex;
		SpikeMonitor * result_monitor_layer2_in;
		SpikeMonitor * result_monitor_layer3;
};


#endif