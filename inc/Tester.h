#ifndef _TESTER_H
#define _TESTER_H

void printInMat(float *input_matrix);

void testMessageQueue();

void testUnrolling(std::string model_name, int num_connection);

void testClustering(std::string model_name, int num_connection, std::vector<int>& dim);

void testAlgorithms(std::string model_name);

void setupNames(std::vector<std::string> &dataset_name, std::vector<std::string> &model_name, std::vector<int> &run_time, int num_controller, int task_id);

#endif