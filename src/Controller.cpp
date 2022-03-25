#include <vector>
#include <iostream>
#include <memory>
#include <condition_variable>
#include <map>
#include <queue>
#include <thread>
#include <fstream>
#include <ctime>
#include <carlsim.h>

#include "Microbrain.h"
#include "Message.h"
#include "MessageQueue.h"
#include "NetworkModel.h"
#include "Controller.h"

void senderFunc(int msg_id, std::string& model_name, std::string& dataset_name, int count, MessageQueue &msg_que) {
	NetworkModel network_model(model_name);
	NetworkInput network_input(dataset_name);

	std::vector <std::vector <std::vector <float> > > &weight = network_model.getWeight();
	for (int i = 0; i < count; ++ i) {
		std::vector <float> input_matrix = network_input.getInputMatrix();
		int output_val = network_input.getOutput();
		// tuple weight, input, output
		Controller::PayloadMatrix payload(weight, input_matrix, output_val);
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
	time_t begin_run, end_run;
	for (int i = 0; i < count; ++ i)
		for (int j = 0; j < num_sender; ++ j) {
			begin_run = clock();
			//std::cout << "receiver " << i << " " << j << std::endl;
			auto msg = msg_que_list[j].get();
			auto& datamsg = dynamic_cast<DataMessage<Controller::PayloadMatrix>&>(*msg);
			Controller::PayloadMatrix payload = datamsg.getPayload();
			microbrain.loadWeight(sim, payload.weight);
			std::cout << "Loaded Weight!" << std::endl;
			float input_cnt = microbrain.loadInput(sim, payload.input_matrix);
			std::cout << "Receiver:" << std::endl << "\tSender ID: " << j << " Order: " << i;
			std::cout << " Expected Value: " << payload.output_val << " Result: " << microbrain.testResult(sim, in, input_cnt) << std::endl;
			end_run = clock();
			std::cout << "Running time: " << (double)(end_run - begin_run) / CLOCKS_PER_SEC << std::endl;
		}
}

Controller::Controller(int num_sender): num_sender(num_sender) {
	//msg_que_list.resize(num_sender);
	msg_que_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		msg_que_list.emplace_back(MessageQueue());
}

void Controller::run(std::string &model_name, std::string &dataset_name, int count, Microbrain &microbrain, CARLsim &sim, PoissonRate &in) {
	std::vector <std::thread> thread_list;
	thread_list.reserve(num_sender);
	for (int i = 0; i < num_sender; ++ i)
		thread_list.emplace_back(std::thread(senderFunc, i, std::ref(model_name), std::ref(dataset_name), count, std::ref(msg_que_list[i])));
	//std::thread t_sender(senderFunc, 0, std::ref(model_name), std::ref(dataset_name), count, std::ref(msg_que));
	std::thread t_receiver(receiverFunc, num_sender, count, std::ref(msg_que_list), std::ref(microbrain), std::ref(sim), std::ref(in));

	for (auto &thd: thread_list)
		thd.join();
	//t_sender.join();
	t_receiver.join();
}

