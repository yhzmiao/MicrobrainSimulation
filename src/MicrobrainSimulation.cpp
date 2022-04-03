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

#include "Microbrain.h"
#include "Message.h"
#include "MessageQueue.h"
#include "Tester.h"
#include "NetworkModel.h"
#include "Controller.h"

int main() {
	// keep track of execution time
	//std::string name = "MNIST_largescale";
	//testUnrolling(name, 64);
	//return 0;

	Stopwatch watch;
	

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	std::vector<std::string> dataset_name = {"MNIST_16", "MNIST_16", "MNIST_16"};
	std::vector<std::string> model_name = {"MNIST_negative", "MNIST_positive", "MNIST_negative"};
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
	PoissonRate in(in_size, false);
	in.setRates(500.0f);
	//microbrain.loadInput(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 0, in);

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	//std::cout << microbrain.testAccuracy(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 100, in) << std::endl;
	// n = 10000 94.42

	// initialize with the number of sender
	Controller controller(3);
	controller.run(model_name, dataset_name, 10, microbrain, sim, in);
	
	//for (int i=0; i<3; i++) {
	//	sim.runNetwork(1,0);
	//}

	// print stopwatch summary

	watch.stop();
	//printInMat(input_matrix);
	//microbrain.getResults();

	return 0;
}
