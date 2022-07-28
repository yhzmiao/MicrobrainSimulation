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
#include <ctime>

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

void testAlgorithms(std::string model_name) {
    time_t begin_unroll, end_unroll, begin_cluster = 0, end_cluster = 0;
    int size_arr[5] = {64, 128, 256, 512, 1024}, num_neuron_before, num_neuron_after, num_cluster = -1;
    std::ofstream ofs("results/Task2.out", std::ios::app);


    for (auto size: size_arr) {
        NetworkModel nm(model_name);
        num_neuron_before = nm.getNeuronSize();
        std::cout << model_name << " " << size << std::endl;
        std::cout << "Unrolling!" << std::endl;

        begin_unroll = clock();
        nm.networkUnrolling(size);
        end_unroll = clock();

        num_neuron_after = nm.getNeuronSize();

        std::cout << "Clustering!" << std::endl;

        std::pair <double, double> utilization = std::make_pair(-1, -1);
        
        // 64 6000
        // 128 8000
        bool to_cluster = true;
        if (size == 64 && num_neuron_after > 15000)
            to_cluster = false;
        if (size == 128 && num_neuron_after > 8000)
            to_cluster = false;
        if (to_cluster) {
            std::vector<int> dim = {size, size / 4};
            begin_cluster = clock();
            num_cluster = nm.networkClustering(dim);
            end_cluster = clock();
            utilization = nm.getUtilization();
        }

        ofs << model_name << " " << size << " " << (double)(end_unroll - begin_unroll) / CLOCKS_PER_SEC << " " << num_neuron_before << " " << num_neuron_after << " " << num_neuron_after - num_neuron_before << " " << (double)(end_cluster - begin_cluster) / CLOCKS_PER_SEC << " " << num_cluster << " " << utilization.first << std::endl;  //" " << utilization.second << std::endl;
    }

    ofs.close();
}

void setupNames(std::vector<std::string> &dataset_name, std::vector<std::string> &model_name, std::vector<int> &run_time, int num_controller, int task_id) {
    if (task_id == 4) {
        dataset_name.resize(num_controller, dataset_name[0]);
        model_name.resize(num_controller, model_name[0]);
        run_time.resize(num_controller, run_time[0]);
        return;
    }

    std::vector<std::string> d_name = {"MNIST_16", "MNIST_16", "MNIST_32", "MNIST_32", "FashionMNIST", "SVHN"};
    std::vector<std::string> m_name = {"MNIST_negative", "MNIST_negative_2", "MNIST_largescale_3", "MNIST_largescale_3_2", "FashionMNIST", "SVHN"};
    std::vector<int> r_time = {100, 200, 400, 400, 400, 400};

    dataset_name.resize(num_controller);
    model_name.resize(num_controller);
    run_time.resize(num_controller);

    for (int i = 0; i < num_controller; ++ i) {
        dataset_name[i] = d_name[i % 6];
        model_name[i] = m_name[i % 6];
        run_time[i] = r_time[i % 6];
    }
}