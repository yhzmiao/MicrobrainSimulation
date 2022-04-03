#ifndef _NETWORKMODEL_H
#define _NETWORKMODEL_H

class Neuron {
	public:
		Neuron(int neuron_id, std::vector<int> output_neuron, std::vector<float> weight);
		void addInput(int i);
		int getNeuronId();
		std::vector<int> getOutput();

		// for unrolling
		int extractInput();
		int getInputSize();
		void updateOutput(int src, int dst);
	private:
		int neuron_id, update_pointer;
		std::queue<int> input_neuron;
		//std::vector<float> weight;
		//std::vector<int> output_neuron;
		std::map <int, float> output_synapse;
};

/*
class SNN {

}
*/

class NetworkModel {
	public:
		NetworkModel(std::string model_name);
		~NetworkModel() = default;

		// get weight by reference
		std::vector <std::vector <std::vector <float> > >& getWeight();
		void networkUnrolling(int num_connection);

		std::vector<Neuron> getNeuronList();

	private:
		bool large_scale;

		std::string model_name;
		std::vector <int> dim;
		std::vector <std::vector <std::vector <float> > > weight; // [0,1] i j

		std::vector <Neuron> neuron_list;
		//std::vector <std::vector <float> > 
};

class NetworkInput {
	public:
		NetworkInput(std::string dataset_name);
		~NetworkInput();

		// get input by reference
		std::vector <float> & getInputMatrix();
		int getOutput();
		std::pair <int, int> getInputInfo();
	private:
		std::ifstream fin_input;
		std::ifstream fin_output;
		int input_dim, input_size;
		std::vector <float> input_matrix;
};

#endif