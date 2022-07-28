// include CARLsim user interface
#include <carlsim.h>
#include <cstdlib>
#include <iostream>

// include stopwatch for timing
#include <stopwatch.h>

// some additional include
#include <memory>
#include <mutex>
#include <map>
#include <queue>
#include <condition_variable>
#include <fstream>
#include <string>
#include <cstdlib>

#include <unistd.h>
#include <getopt.h>

#include "Microbrain.h"
#include "Message.h"
#include "MessageQueue.h"
#include "Tester.h"
#include "NetworkModel.h"
#include "Controller.h"



int main(int argc, char **argv) {
	
	//std::string name = "MNIST_largescale";
	//std::vector<int> tmp_dim = {256, 64};
	//testClustering(name, 256, tmp_dim);
	//testUnrolling(name, 256);
	
	//std::string name = "Tiny";
	//std::vector<int> tmp_dim = {3, 3};
	//testClustering(name, 3, tmp_dim);
	//testUnrolling(name, 3); 
	//return 0;
	

	int task_id = 0, loop_time = 1000, controller_size = 1;
	std::vector <int> run_time = {100};

	std::vector<std::string> dataset_name = {"MNIST_32"};
	std::vector<std::string> model_name = {"MNIST_largescale_3"};

	int opt;
	char *optstring = (char *)"t:r:d:m:l:c:";
	int option_index = 0;
	static struct option long_options[] = {
		{"taskid", optional_argument, NULL, 't'},
		{"runtime", optional_argument, NULL, 'r'},
		{"dataset", optional_argument, NULL, 'd'},
		{"model", optional_argument, NULL, 'm'},
		{"looptime", optional_argument, NULL, 'l'},
		{"controller", optional_argument, NULL, 'c'}
	};

	while ( (opt = getopt_long(argc, argv, optstring, long_options, &option_index)) != -1) {
		switch (opt) {
			case 't':
				task_id = atoi(optarg);
				break;
			case 'r':
				run_time[0] = atoi(optarg);
				break;
			case 'd':
				dataset_name[0] = optarg;
				break;
			case 'm':
				model_name[0] = optarg;
				break;
			case 'l':
				loop_time = atoi(optarg);
				break;
			case 'c':
				controller_size = atoi(optarg);
				break;
			default:
				std::cout << "Error" << std::endl;
				break;
		}
	}

	if(task_id == 2) {
		testAlgorithms(model_name[0]);
		return 0;
	}
	if(task_id == 4 || task_id == 5)
		setupNames(dataset_name, model_name, run_time, controller_size, task_id);

	std::cout << "Processed commands!" << std::endl;
	// keep track of execution time
	Stopwatch watch;
	

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	//std::vector<std::string> dataset_name = {"MNIST_32"};//, "MNIST_16", "MNIST_16"};
	//std::vector<std::string> model_name = {"MNIST_largescale_2"};//, "MNIST_positive", "MNIST_negative"};
	std::vector <int> dim = {256, 64, 10};
	bool single_neuron_group = false;

	CARLsim sim("microbrain demo", CPU_MODE, USER, numGPUs, randSeed);

	FILE * fp;
	float input_matrix[NUM_NEURON_LAYER1];

	Microbrain microbrain;
	microbrain.setupNeurons(sim);
	microbrain.setupConnections(sim);
		
	sim.setConductances(false);

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	
	//microbrain.loadWeight(sim, model_name, dim);
	
	int in_size = NUM_NEURON_LAYER1;
	//std::vector<PoissonRate> in_list(in_size, PoissonRate(1, false));
	//in.setRates(500.0f);
	PoissonRate in(in_size, false);
	//microbrain.loadInput(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 0, in);

	// ---------------- RUN STATE -------------------
	//watch.lap("runNetwork");

	//std::cout << microbrain.testAccuracy(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 100, in) << std::endl;
	// n = 10000 94.42

	// initialize with the number of sender

	//Task 1: running time vs accuracy
	
	Controller controller(controller_size);
	controller.run(model_name, dataset_name, loop_time, microbrain, sim, in, run_time, task_id);

	
	//for (int i=0; i<3; i++) {
	//	sim.runNetwork(1,0);
	//}

	// print stopwatch summary

	watch.stop();
	//printInMat(input_matrix);
	//microbrain.getResults();

	return 0;
}
