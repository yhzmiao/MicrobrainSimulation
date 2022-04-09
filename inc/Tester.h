#ifndef _TESTER_H
#define _TESTER_H

void printInMat(float *input_matrix);

void testMessageQueue();

void testUnrolling(std::string model_name, int num_connection);

void testClustering(std::string model_name, int num_connection, std::vector<int>& dim);

#endif