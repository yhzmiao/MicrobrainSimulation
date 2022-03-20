#ifndef _NETWORKMODEL_H
#define _NETWORKMODEL_H

class NetworkModel {
	public:
		NetworkModel(std::string& model_name);
		~NetworkModel() = default;

		// get weight by reference
		std::vector <std::vector <std::vector <float> > >& getWeight();
	private:
		std::string model_name;
		std::vector <int> dim;
		std::vector <std::vector <std::vector <float> > > weight1; // [0,1] i j
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
}

#endif