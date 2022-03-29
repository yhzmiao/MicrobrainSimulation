#ifndef _NETWORKMODEL_H
#define _NETWORKMODEL_H

class Neuron {
	public:
		Neuron(int neuron_id, std::vector<int> output_neuron, std::vector<float> weight): neuron_id(neuron_id), output_neuron(output_neuron), weight(weight){}
		void addOutput(int i);
	private:
		int neuron_id;
		std::vector<int> input_neuron;
		std::vector<float> weight;
		std::vector<int> output_neuron;
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
	private:
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