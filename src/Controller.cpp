#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>
#include <ctime>
#include <unordered_map>
#include <carlsim.h>

#include "Microbrain.h"
#include "Message.h"
#include "MessageQueue.h"
#include "NetworkModel.h"
#include "Controller.h"
#include "Strategy.h"


void senderFunc(int msg_id, std::string& model_name, std::string& dataset_name, int count, MessageQueue &msg_que) {
	//NetworkModel network_model(model_name);
	NetworkInput network_input(dataset_name);

	//std::vector <std::vector <std::vector <float> > > &weight = network_model.getWeight();
	for (int i = 0; i < count; ++ i) {
		std::vector <float> input_matrix = network_input.getInputMatrix();
		int output_val = network_input.getOutput();
		// tuple weight, input, output
		Controller::PayloadMatrix payload(model_name, input_matrix, output_val, clock());
		/*std::cout << "sender" << msg_id << " " << i << std::endl;
		for (int i = 0; i < 64; ++ i)
			std::cout << payload.weight[0][0][i] << " ";
		std::cout << std::endl;
		*/
		msg_que.put(DataMessage <Controller::PayloadMatrix>(msg_id, payload));
	}
}

void receiverFunc(int num_sender, int count, std::vector <MessageQueue> &msg_que_list, Microbrain &microbrain, CARLsim &sim, PoissonRate &in) {
	// round robin
	std::vector <NetworkModel> model_list;
	std::unordered_map<std::string, int> model_map;
	std::unordered_map<std::string, int>::iterator it;
	model_list.reserve(num_sender);

	int total_query = num_sender * count;
	std::vector <int> rest_query(num_sender, count);

	time_t begin_run, end_run;
	int correct_cnt = 0, total_cnt = 0;

	// create strategy and initilization
	StrategyManager strategy_manager(new RoundRobinStrategy);
	std::vector <QueryInformation> query_information_list(num_sender, QueryInformation());
	std::vector <RunningTask> task_list(1, RunningTask(-1, 0));

	std::vector<std::pair<int, int> > cluster_input_matrix;
	std::vector<int> spike_time;
	//std::vector <float> input_matrix;

	// loop until finished all query
	while (total_query) {
		strategy_manager.getSchedule(query_information_list, task_list);
		int q_id = task_list[task_list.size() - 1].query_id;
		QueryInformation &q = query_information_list[q_id];
		std::cout << "Execution on sender " << q_id << std::endl;

		if (q.model_id == -1 || (q.cluster_id == model_list[q.model_id].getClusterSize() && rest_query[q_id])) {
			int model_id = 0, num_cluster = 0;

			auto msg = msg_que_list[q_id].get();
			auto& datamsg = dynamic_cast<DataMessage<Controller::PayloadMatrix>&>(*msg);
			
			float in_map = true;

			Controller::PayloadMatrix payload = datamsg.getPayload();
			it = model_map.find(payload.model_name);

			// if find this in map
			if (it == model_map.end()) {
				in_map = false;
				model_id = model_list.size();
				model_map.insert(make_pair(payload.model_name, model_id));
				NetworkModel network_model(payload.model_name);

				//todo: add some algorithm here
				network_model.networkUnrolling(256);
				std::vector<int> tmp_dim = {256, 64};
				num_cluster = network_model.networkClustering(tmp_dim);

				model_list.emplace_back(network_model);
			}

			else {
				model_id = it->second;
				num_cluster = model_list[model_id].getClusterSize();
			}

			q.setValue(model_id, 0, num_cluster, payload.output_val, payload.time_stamp, in_map, model_list[model_id].getNeuronSize());
			model_list[model_id].setInputMatrix(payload.input_matrix, q);
			
			std::cout << "Setup sender information!" << std::endl;
		}


		//int input_cnt = model_list[model_id].setInputMatrix(q);
		//std::cout << q.model_id << " " << q.cluster_id << std::endl;
		microbrain.loadWeight(sim, model_list[q.model_id], q.model_id, q.cluster_id, q.in_map);
		if (!q.in_map)
			microbrain.saveWeightPointer(sim, q.model_id);
		cluster_input_matrix = model_list[q.model_id].getInputMatrix(q);
		spike_time = microbrain.testResult(sim, cluster_input_matrix, in, model_list[q.model_id].getRunningTime());

		model_list[q.model_id].updateInput(q.cluster_id, spike_time, q);

		q.update();
		if (q.cluster_id == model_list[q.model_id].getClusterSize()) {
			int test_result = model_list[q.model_id].getResult(q);
			
			std::cout << "Sender ID: " << q_id << std::endl ;
			std::cout << "Expected Value: " << q.output_val << std::endl;
			std::cout << "Output Value: " << test_result << std::endl;

			rest_query[q_id] --;
			total_query --;
			if (!rest_query[q_id])
				q.weight = -1;
		}
	}

	/*
	for (int i = 0; i < count; ++ i) {
		std::cout << "Quest ID: " << i << std::endl;
		for (int j = 0; j < num_sender; ++ j) {
			begin_run = clock();
			//std::cout << "receiver " << i << " " << j << std::endl;
			auto msg = msg_que_list[j].get();
			auto& datamsg = dynamic_cast<DataMessage<Controller::PayloadMatrix>&>(*msg);
			Controller::PayloadMatrix payload = datamsg.getPayload();
			it = model_map.find(payload.model_name);
			int model_id = 0, num_cluster = 0;
			double loading_time = 0;
			float in_map = false;
			if (it == model_map.end()) {
				model_id = model_list.size();
				model_map.insert(make_pair(payload.model_name, model_id));
				NetworkModel network_model(payload.model_name);

				//todo: add some algorithm here
				network_model.networkUnrolling(256);
				std::vector<int> tmp_dim = {256, 64};
				num_cluster = network_model.networkClustering(tmp_dim);

				model_list.emplace_back(network_model);
				
				
				//for (int cluster_id = 0; cluster_id < num_cluster; ++ cluster_id) {
				//	loading_time = microbrain.loadWeight(sim, model_list[model_id].getWeight());
				//	microbrain.saveWeightPointer(sim, model_id);
				//}
				
			}
			else {
				model_id = it->second;
				//loading_time = microbrain.loadWeight(sim, model_id, 0);
				num_cluster = model_list[model_id].getClusterSize();
				in_map = true;
			}

			int input_cnt = model_list[model_id].setInputMatrix(payload.input_matrix);
			
			std::vector<std::pair<int, int> > cluster_input_matrix;
			for (int cluster_id = 0; cluster_id < num_cluster; ++ cluster_id) {
				std::cout << "Now running on cluster " << cluster_id << std::endl;
				loading_time += microbrain.loadWeight(sim, model_list[model_id], model_id, cluster_id, in_map);
				if (!in_map)
					microbrain.saveWeightPointer(sim, model_id);
				cluster_input_matrix = model_list[model_id].getInputMatrix(cluster_id); // todo
				//microbrain.loadInput(sim, cluster_input_matrix);
				//std::cout << "Loaded input matrix!" << std::endl;

				std::vector<int> spike_time = microbrain.testResult(sim, cluster_input_matrix, in, model_list[model_id].getRunningTime());
				//for (int i = 0; i < spike_time.size(); ++ i) {
				//	std::cout << i << " " << spike_time[i] << std::endl;
				//}
				model_list[model_id].updateInput(cluster_id, spike_time); // todo
			}

			int test_result = model_list[model_id].getResult(); // todo
			
			//loading_time = microbrain.loadWeight(sim, model_list[model_id].getWeight());
			std::cout << std::endl << "Loaded Weight!" << std::endl;
			std::cout << "Receiver:" << std::endl << "\tSender ID: " << j << " Order: " << i;
			std::cout << " Expected Value: " << payload.output_val << std::endl;

			std::cout << std::endl << "Total Input: " << input_cnt ;
			std::cout << std::endl << "Running Result: " << test_result;
			total_cnt ++;
			if (test_result == payload.output_val) {
				std::cout << " (correct)" << std::endl;
				correct_cnt ++;
			}
			else
				std::cout << " (wrong)" << std::endl;
			end_run = clock();
			double running_time = (double)(end_run - begin_run) / CLOCKS_PER_SEC;
			std::cout << "Running time: " << running_time << " (" << loading_time << " + " << 0.1 * num_cluster << " + " << running_time - loading_time - 0.1 << ")"<< std::endl;
			std::cout << "Accuracy: " << correct_cnt << "/" << total_cnt << " (" << (double)correct_cnt / (double)total_cnt << ")" << std::endl;
			std::cout << "==========================================================================" << std::endl;
			//microbrain.recoverInput(sim, payload.input_matrix);
		}
	}
	*/
}

Controller::Controller(int num_sender): num_sender(num_sender) {
	//msg_que_list.resize(num_sender);
	msg_que_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		msg_que_list.emplace_back(MessageQueue());
}

void Controller::run(std::vector<std::string> &model_name, std::vector<std::string> &dataset_name, int count, Microbrain &microbrain, CARLsim &sim, PoissonRate &in) {
	//microbrain.initInputWeightPointer(sim);

	std::vector <std::thread> thread_list;
	thread_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		thread_list.emplace_back(std::thread(senderFunc, i, std::ref(model_name[i]), std::ref(dataset_name[i]), count, std::ref(msg_que_list[i])));
	//std::thread t_sender(senderFunc, 0, std::ref(model_name), std::ref(dataset_name), count, std::ref(msg_que));
	std::thread t_receiver(receiverFunc, num_sender, count, std::ref(msg_que_list), std::ref(microbrain), std::ref(sim), std::ref(in));

	for (auto &thd: thread_list)
		thd.join();
	//t_sender.join();
	t_receiver.join();
}

