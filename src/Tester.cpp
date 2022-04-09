#include <iostream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <thread>
#include <fstream>
#include <vector>
#include <carlsim.h>

#include "Message.h"
#include "MessageQueue.h"
#include "Microbrain.h"
#include "NetworkModel.h"

void printInMat(float *input_matrix) {
	for (int i = 0; i < NUM_NEURON_LAYER1; ++ i) {
		if (i % 16 == 0)
			std::cout << std::endl;
		if (input_matrix[i] < 0.5)
			std::cout << "  ";
		else
			std::cout << "* ";
	}
	std::cout << std::endl;
}

void testMessageQueue() {
	int N = 500;
	MessageQueue msg_que;

    auto sender = [](int msgId, int count, MessageQueue& q) {
		std::vector <int> a = {1, 2, 3};
        for (int i = 0; i < count; ++i)
            q.put(DataMessage<std::vector<int> >(msgId, a));
    };

    auto receiver = [](int count, MessageQueue& q) {

        for (int i = 0; i < count; ++i) {
            auto m = q.get();
            auto& dm = dynamic_cast<DataMessage<std::vector<int> >&>(*m);

            std::cout << dm.getMessageId() << " " << dm.getPayload().size() << std::endl;
        }
    };

    std::thread t1(sender, 1, N, std::ref(msg_que));
    std::thread t2(sender, 2, N, std::ref(msg_que));
    std::thread t3(receiver, 2 * N, std::ref(msg_que));
    t1.join();
    t2.join();
    t3.join();
}

void testUnrolling(std::string model_name, int num_connection) {
    NetworkModel nm(model_name);
    nm.networkUnrolling(num_connection);
    std::vector<Neuron> neuron_list = nm.getNeuronList();

    std::cout << std::endl << neuron_list.size() << std::endl;

    for (auto &neuron: neuron_list) {
        std::cout << neuron.getNeuronId() << " ";
        std::vector<int> output_neuron = neuron.getOutput();
        //std::cout << "size " << output_neuron.size() << std::endl;
        for (auto out: output_neuron)
            std::cout << out << " ";
        std::cout << std::endl;
    }
    
}

void testClustering(std::string model_name, int num_connection, std::vector<int>& dim) {
    NetworkModel nm(model_name);
    nm.networkUnrolling(num_connection);
    std::vector<Neuron> neuron_list = nm.getNeuronList();

    //std::cout << std::endl << neuron_list.size() << std::endl;
    
    for (auto &neuron: neuron_list) {
        std::cout << neuron.getNeuronId() << " ";
        std::vector<int> output_neuron = neuron.getOutput();
        for (auto out: output_neuron)
            std::cout << out << " ";
        std::cout << std::endl;
    }
    

    std::cout << "Start Clustering!!" << std::endl;

    int num_cluster = nm.networkClustering(dim);
    std::cout << "Total number of clusters: "<< num_cluster << std::endl;
}