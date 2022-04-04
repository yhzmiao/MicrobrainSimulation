#ifndef _NETWORKMODEL_H
#define _NETWORKMODEL_H

#define ITER_TIME 50
#define RAMDOM_TIME 200

class Neuron {
	public:
		Neuron(int neuron_id, std::vector<int> output_neuron, std::vector<float> weight);
		void addInput(int i);
		int getNeuronId();
		std::vector<int>& getOutput();

		// for unrolling
		int extractInput();
		int getInputSize();
		void updateOutput(int src, int dst);

		// for clustering
		std::vector<int>& getInput();
		void clusteringUpdateSet(std::set<int> &input_set, std::set<int> &output_set);
		std::vector<int> getCluster(std::vector<std::set<int> >& neuron_set_list);
		void ClusteringRemoveDummy(std::set<int> &input_set);
		
		bool input_settled;

	private:
		int neuron_id, update_pointer;
		std::queue<int> input_neuron;
		std::vector<int> input_neuron_list;
		//std::vector<float> weight;
		std::map <int, float> output_synapse;
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
		
		void networkUnrolling(int num_connection);
		int networkClustering(std::vector<int> dim);

		std::vector<Neuron> getNeuronList();

	private:
		bool large_scale;

		std::string model_name;
		std::vector <int> dim;
		std::vector <std::vector <std::vector <float> > > weight; // [0,1] i j

		std::vector <Neuron> neuron_list;
		std::vector <std::vector <pair <int, int> > > neuron_cluster_list;
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










