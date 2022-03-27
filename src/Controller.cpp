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


void senderFunc(int msg_id, std::string& model_name, std::string& dataset_name, int count, MessageQueue &msg_que) {
	//NetworkModel network_model(model_name);
	NetworkInput network_input(dataset_name);

	//std::vector <std::vector <std::vector <float> > > &weight = network_model.getWeight();
	for (int i = 0; i < count; ++ i) {
		std::vector <float> input_matrix = network_input.getInputMatrix();
		int output_val = network_input.getOutput();
		// tuple weight, input, output
		Controller::PayloadMatrix payload(model_name, input_matrix, output_val);
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

	time_t begin_run, end_run;
	for (int i = 0; i < count; ++ i)
		for (int j = 0; j < num_sender; ++ j) {
			begin_run = clock();
			//std::cout << "receiver " << i << " " << j << std::endl;
			auto msg = msg_que_list[j].get();
			auto& datamsg = dynamic_cast<DataMessage<Controller::PayloadMatrix>&>(*msg);
			Controller::PayloadMatrix payload = datamsg.getPayload();
			it = model_map.find(payload.model_name);
			int model_id = 0;
			if (it == model_map.end()) {
				model_id = model_list.size();
				model_map.insert(make_pair(payload.model_name, model_id));
				model_list.emplace_back(NetworkModel(payload.model_name));
			}
			else
				model_id = it->second;

			double loading_time = microbrain.loadWeight(sim, model_list[model_id].getWeight());
			std::cout << "Loaded Weight!" << std::endl;
			float input_cnt = microbrain.loadInput(sim, payload.input_matrix);
			std::cout << "Receiver:" << std::endl << "\tSender ID: " << j << " Order: " << i;
			std::cout << " Expected Value: " << payload.output_val << std::endl;
			int test_result = microbrain.testResult(sim, in, input_cnt);
			std::cout << std::endl << "Running Result: " << test_result;
			if (test_result == payload.output_val)
				std::cout << " (correct)" << std::endl;
			else
				std::cout << " (wrong)" << std::endl;
			end_run = clock();
			double running_time = (double)(end_run - begin_run) / CLOCKS_PER_SEC;
			std::cout << "Running time: " << running_time << " (" << loading_time << " + 0.1 + " << running_time - loading_time - 0.1 << ")"<< std::endl;
			std::cout << "==========================================================================" << std::endl;
		}
}

Controller::Controller(int num_sender): num_sender(num_sender) {
	//msg_que_list.resize(num_sender);
	msg_que_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		msg_que_list.emplace_back(MessageQueue());
}

void Controller::run(std::vector<std::string> &model_name, std::vector<std::string> &dataset_name, int count, Microbrain &microbrain, CARLsim &sim, PoissonRate &in) {
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

