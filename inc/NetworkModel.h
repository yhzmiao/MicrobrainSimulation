#ifndef _NETWORKMODEL_H
#define _NETWORKMODEL_H

#include <set>
#include <map>
#include <queue>
#include <ctime>
#include <random>

#define ITER_TIME 1
#define RECLUSTER_TIME 10
#define RANDOM_TIME 500
//#define RUN_TIME 400

struct RunningTask{
	int query_id;
	time_t time_stamp;

	RunningTask() {}
	RunningTask(int query_id, time_t time_stamp): query_id(query_id), time_stamp(time_stamp) {}
	RunningTask(const RunningTask &rt): query_id(rt.query_id), time_stamp(rt.time_stamp) {}
};

struct QueryInformation{
	int model_id;
	int cluster_id;
	int weight;
	int output_val;
	int run_time;
	time_t time_stamp;
	bool in_map;
	//std::vector <float> input_matrix;
	std::vector <std::pair<int, int>> spike_rate;

	QueryInformation() {model_id = -1; weight = 1;}
	//QueryInformation(int model_id, int cluster_id, int weight, time_t time_stamp, std::vector <int> spike_rate): model_id(model_id), cluster_id(cluster_id), weight(weight), time_stamp(time_stamp), spike_rate(spike_rate) {}
	//QueryInformation(const QueryInformation &qi): model_id(qi.model_id), cluster_id(qi.cluster_id), weight(qi.weight), time_stamp(qi.time_stamp), spike_rate(qi.spike_rate) {}

	void setValue(int m_id, int c_id, int w, int o_v, int rt, time_t ts, bool im, int spike_size);
	void update();
	void update_ts(time_t ts);
};

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
		float getWeight(int output_id);

		// for mapping
		void setInputRate(int rate, bool ex = true);
		std::pair<int, int> getInputRate();
		
		bool input_settled;
		bool lazy_update;

	private:
		int neuron_id, update_pointer, input_rate_ex, input_rate_in;
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
		NetworkModel(std::string model_name, int ls_rt = 100);
		//NetworkModel(const NetworkModel &nm);
		~NetworkModel() = default;

		// get weight by reference
		std::vector <std::vector <std::vector <float> > >& getWeight();
		
		void networkUnrolling(int num_connection);
		int networkClustering(std::vector<int> dim);

		void clusteringUpdateSet(std::set<int> &input_set, std::set<int> &output_set);
		void ClusteringRemoveDummy(std::set<int> &input_set);
		std::vector<std::pair<int, int>> getCluster(std::vector<std::set<int> >& neuron_set_list, std::vector<int>& dim, std::default_random_engine &rd);

		std::vector<Neuron> getNeuronList();
		std::vector<std::pair<int, int> >& getCluster(int cluster_id);
		void setClusterWeight(int cluster_id, std::vector <std::vector <std::vector<float> > >& weight);
		int getClusterSize();
		int getNeuronSize();

		// for mapping clusters
		int setInputMatrix(std::vector<float>& input_matrix, QueryInformation &q);
		std::vector<std::pair<int, int> > getInputMatrix(QueryInformation &q);
		void updateInput(int cluster_id, std::vector<int>& spike_time, QueryInformation &q);
		std::pair<int, int> getResult(QueryInformation &q);
		std::pair<double, double> getUtilization();
		int getRunningTime();

		std::string getModelName();

	private:
		bool large_scale;

		int run_time = 100;

		std::string model_name;
		std::vector <int> dim;
		std::vector <int> mb_dim;
		std::vector <std::vector <std::vector <float> > > weight; // [0,1] i j

		std::vector <Neuron> neuron_list;
		std::vector <std::vector <std::pair <int, int> > > neuron_cluster_list;
		//std::vector <std::vector <float> > 

		// for creating instance
		//std::vector <std::pair<int, int> > model_condition_list; //first: model_id, second: cluster;
		//std::vector <std::vector <float> > model_rate_list;
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

