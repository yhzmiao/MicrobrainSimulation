
/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
* CARLsim5: HK, JX, KC
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

// include CARLsim user interface
#include <carlsim.h>
#include <cstdlib>
#include <iostream>

// include stopwatch for timing
#include <stopwatch.h>
#include "mubrain.h"

int main() {
	// keep track of execution time
	Stopwatch watch;
	

	// ---------------- CONFIG STATE -------------------
	
	// create a network on GPU
	int numGPUs = 1;
	int randSeed = 42;
	std::string dataset_name = "MNIST";
	std::string model_name = "MNIST_positive";
	std::vector <int> dim = {256, 64, 10};
	bool single_neuron_group = false;

	CARLsim sim("microbrain demo", CPU_MODE, USER, numGPUs, randSeed);

	FILE * fp;
	float input_matrix[NUM_NEURON_LAYER1];

	Mubrain mubrain(true, false, single_neuron_group);
	mubrain.setupNeurons(sim);
	mubrain.setupConnections(sim);
		
	sim.setConductances(false);

	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

	
	mubrain.loadWeight(sim, model_name, dim);
	
	int in_size = single_neuron_group ? 1 : NUM_NEURON_LAYER1;
	PoissonRate in(in_size, false);
	in.setRates(500.0f);
	//mubrain.loadInput(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 0, in);

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

	std::cout << mubrain.testAccuracy(sim, dataset_name, input_matrix, NUM_NEURON_LAYER1, 100, in) << std::endl;
	// n = 10000 94.42
	
	//for (int i=0; i<3; i++) {
	//	sim.runNetwork(1,0);
	//}

	// print stopwatch summary

	watch.stop();
	//printInMat(input_matrix);
	//mubrain.getResults();

	return 0;
}
