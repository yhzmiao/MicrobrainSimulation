#ifndef _CONTROLLER_H
#define _CONTROLLER_H

//todo: maybe some complicated algorithm
class Controller {
	public:
		Controller(int num_sender);
		~Controller() = default;

		void run(std::string &model_name, std::string &dataset_name, int count);

		struct PayloadMatrix {
			std::vector <std::vector <std::vector <float> > > &weight;
			std::vector <float> &input_matrix;
			int output_val;
			PayloadMatrix(
				std::vector <std::vector <std::vector <float> > > &weight, 
				std::vector <float> &input_matrix, 
				int output_val):
				weight(weight),
				input_matrix(input_matrix),
				output_val(output_val){}
		};
	private:
		std::vector <MessageQueue> msg_que_list;
		//MessageQueue msg_que;
		int num_sender;
};

void sender(int msg_id, std::string &model_name, std::string &dataset_name, int count, MessageQueue &msg_que);
void receiver(int num_sender, int count, std::vector <MessageQueue> &msg_que_list);

#endif