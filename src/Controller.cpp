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

void receiverFunc(int num_sender, int count, std::vector <MessageQueue> &msg_que_list, Microbrain &microbrain, CARLsim &sim, PoissonRate &in, std::vector<int> &run_time, int task_id) {
	// round robin
	std::vector <NetworkModel> model_list;
	std::unordered_map<std::string, int> model_map;
	std::unordered_map<std::string, int>::iterator it;
	model_list.reserve(num_sender);

	int total_query = num_sender * count;
	std::vector <int> rest_query(num_sender, count);

	time_t begin_run, end_run, begin_query, end_query;
	int correct_cnt = 0, total_cnt = 0;

	// create strategy and initilization
	//StrategyManager strategy_manager(new RoundRobinStrategy);
	//StrategyManager strategy_manager(new HRRSStrategy);
	//StrategyManager strategy_manager(new RandomStrategy);
	//StrategyManager strategy_manager(new WeightedRandomStrategy);
	StrategyManager strategy_manager(new FCFSStrategy);
	std::vector <QueryInformation> query_information_list(num_sender, QueryInformation());
	std::vector <RunningTask> task_list(1, RunningTask(-1, clock()));
	// waiting time, start running time, cluster start waiting ,start cluster time
	std::vector <std::vector <time_t>> time_stamp_list(num_sender);
	// total waiting time, total execution time, cluster waiting time, total cluster execution time
	std::vector <std::vector <double>> total_time(num_sender);
	// for update time_stamp
	std::vector <time_t> time_stamp_to_update(num_sender, 0);

	for (auto &ts: time_stamp_list)
		ts.resize(4, clock());
	for (auto &tt: total_time)
		tt.resize(4, 0);

	std::vector<std::pair<int, int> > cluster_input_matrix;
	std::vector<int> spike_time;
	//std::vector <float> input_matrix;
	double running_time = 0, waiting_time = 0;
	int query_cnt = 0, active_spike = 0;
	long long total_spike = 0, spike_transformed = 0;

	// loop until finished all query
	while (total_query) {
		begin_query = clock();
		strategy_manager.getSchedule(query_information_list, task_list);
		int q_id = task_list[task_list.size() - 1].query_id;
		QueryInformation &q = query_information_list[q_id];
		std::cout << "Execution on sender " << q_id << " " << total_cnt << std::endl;

		if (q.model_id == -1 || (q.cluster_id == model_list[q.model_id].getClusterSize() && rest_query[q_id])) {
			int model_id = 0, num_cluster = 0, r_time = 0;

			auto msg = msg_que_list[q_id].get();
			auto& datamsg = dynamic_cast<DataMessage<Controller::PayloadMatrix>&>(*msg);

			double in_map = true;

			Controller::PayloadMatrix payload = datamsg.getPayload();
			it = model_map.find(payload.model_name);

			// if find this in map
			if (it == model_map.end()) {
				in_map = false;
				model_id = model_list.size();
				r_time = run_time[q_id];
				model_map.insert(make_pair(payload.model_name, model_id));
				NetworkModel network_model(payload.model_name, r_time);

				//todo: add some algorithm here
				network_model.networkUnrolling(256);
				std::cout << network_model.getNeuronSize() << std::endl;
				std::vector<int> tmp_dim = {256, 64};
				num_cluster = network_model.networkClustering(tmp_dim);

				model_list.emplace_back(network_model);
			}

			else {
				model_id = it->second;
				num_cluster = model_list[model_id].getClusterSize();
				r_time = model_list[model_id].getRunningTime();
			}

			q.setValue(model_id, 0, num_cluster, payload.output_val, r_time, payload.time_stamp, in_map, model_list[model_id].getNeuronSize());
			time_stamp_to_update[q_id] = payload.time_stamp;
			model_list[model_id].setInputMatrix(payload.input_matrix, q);
			total_time[q_id][0] += (double)(clock() - time_stamp_list[q_id][0]) / CLOCKS_PER_SEC;
			time_stamp_list[q_id][0] = clock();
			time_stamp_list[q_id][1] = clock();
			
			std::cout << "Setup sender information!" << std::endl;
		}

		// if it's the last cluster
		//if (cluster_id == model_list[q.model_id].getClusterSize() - 1)
		//	time_stamp_list[q_id][0] = clock();
		total_time[q_id][2] += (double)(clock() - time_stamp_list[q_id][2]) / CLOCKS_PER_SEC;
		time_stamp_list[q_id][2] = clock();
		time_stamp_list[q_id][3] = clock();
		//int input_cnt = model_list[model_id].setInputMatrix(q);
		std::cout << q_id << " " << q.cluster_id << std::endl;
		microbrain.loadWeight(sim, model_list[q.model_id], q.model_id, q.cluster_id, q.in_map);
		if (!q.in_map)
			microbrain.saveWeightPointer(sim, q.model_id);
		cluster_input_matrix = model_list[q.model_id].getInputMatrix(q);
		for (auto &c: cluster_input_matrix)
			total_spike += c.first + c.second;
		begin_run = clock();
		spike_time = microbrain.testResult(sim, cluster_input_matrix, in, model_list[q.model_id].getRunningTime());
		end_run = clock();
		spike_transformed += spike_time[spike_time.size() - 1];
		total_time[q_id][3] += (double)(clock() - time_stamp_list[q_id][3]) / CLOCKS_PER_SEC;

		model_list[q.model_id].updateInput(q.cluster_id, spike_time, q);

		q.update();
		if (q.cluster_id == model_list[q.model_id].getClusterSize()) {
			q.update_ts(time_stamp_to_update[q_id]);
			total_time[q_id][1] += (double)(clock() - time_stamp_list[q_id][1]) / CLOCKS_PER_SEC;
			std::pair<int, int> temp_result = model_list[q.model_id].getResult(q);
			int test_result = temp_result.first;
			active_spike += temp_result.second;
			
			std::cout << "Sender ID: " << q_id << std::endl ;
			std::cout << "Expected Value: " << q.output_val << std::endl;
			std::cout << "Output Value: " << test_result << std::endl;

			total_cnt ++;
			correct_cnt += q.output_val == test_result;
			std::cout << "Accuracy: " << (double)correct_cnt * 100.0 / total_cnt << "% (" << correct_cnt << "/" << total_cnt << ")" << std::endl;

			rest_query[q_id] --;
			total_query --;
			if (!rest_query[q_id])
				q.weight = -1;
			query_cnt ++;
		}
		end_query = clock();
		running_time += (double)(end_run - begin_run) / CLOCKS_PER_SEC;
		waiting_time += (double)(end_query - begin_query) / CLOCKS_PER_SEC;
	}

	if (task_id == 1) {
		std::ofstream ofs("results/Task1.out", std::ios::app);
		ofs << run_time[0] << " " << (double)correct_cnt * 100.0 / total_cnt << std::endl;
		ofs.close();
	}

	if (task_id == 3) {
		std::ofstream ofs("results/Task3.out", std::ios::app);
		ofs << model_list[0].getModelName() << " " << model_list[0].getClusterSize() << " " << model_list[0].getUtilization().first << " " << query_cnt << " " << (double)correct_cnt * 100.0 / total_cnt << " " << running_time / query_cnt << " " << waiting_time / query_cnt << " " << active_spike << " " << total_spike << " " << spike_transformed << std::endl;
		ofs.close();
	}

	if (task_id == 4 || task_id == 5) {
		std::ofstream ofs("results/Task4.out", std::ios::app);
		
		for (int i = 0; i < num_sender; ++ i) {
			int mid = query_information_list[i].model_id;
			ofs << i << " " << model_list[mid].getModelName() << " " << model_list[mid].getClusterSize() << " " << total_time[i][0] << " " << total_time[i][1] << " " << total_time[i][2] << " " << total_time[i][3] << std::endl;
		}

		ofs.close();
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

void Controller::run(std::vector<std::string> &model_name, std::vector<std::string> &dataset_name, int count, Microbrain &microbrain, CARLsim &sim, PoissonRate &in, std::vector<int> &run_time, int task_id) {
	//microbrain.initInputWeightPointer(sim);

	std::vector <std::thread> thread_list;
	thread_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		thread_list.emplace_back(std::thread(senderFunc, i, std::ref(model_name[i]), std::ref(dataset_name[i]), count, std::ref(msg_que_list[i])));
	//std::thread t_sender(senderFunc, 0, std::ref(model_name), std::ref(dataset_name), count, std::ref(msg_que));
	std::thread t_receiver(receiverFunc, num_sender, count, std::ref(msg_que_list), std::ref(microbrain), std::ref(sim), std::ref(in), std::ref(run_time), task_id);

	for (auto &thd: thread_list)
		thd.join();
	//t_sender.join();
	t_receiver.join();
}

