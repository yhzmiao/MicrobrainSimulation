#include <random>
#include <vector>
#include <ctime>

#include "Strategy.h"

void RoundRobinStrategy::schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) {
	int target = task_list[task_list.size() - 1].query_id;

	while (true) {
		target = (target + 1) % query_information_list.size();
		if (query_information_list[target].weight > 0)
			break;
	}

	task_list.emplace_back(new RunningTask(target, clock()));
}

void RandomStrategy::schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) {
	std::random_device rd;
	std::uniform_int_distribution<int> ud(0, query_information_list.size() - 1);

	int target = ud(rd);
	while (query_information_list[target].weight <= 0)
		target = ud(rd);
	
	task_list.emplace_back(new RunningTask(target, clock()));
}

void WeightedRandomStrategy::schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) {
	int sum_weight = 0;
	for(auto &q: query_information_list)
		if (q.weight > 0)
			sum_weight += q.weight;

	std::random_device rd;
	std::uniform_int_distribution<int> ud(1, sum_weight);
	
	int target_weight = ud(rd), target = 0;
	for (int target = 0; target < query_information_list.size(); ++ target)
		if (query_information_list[target].weight > 0) {
			target_weight -= query_information_list[target].weight;
			if(target_weight <= 0)
				break;
		}
	
	task_list.emplace_back(new RunningTask(target, clock()));
}

void FCFSStrategy::schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) {
	int target = -1;

	for(int i = 0; i < query_information_list.size(); ++ i)
		if (query_information_list[i].weight > 0)
			if (target == -1 || query_information_list[target].time_stamp > query_information_list[i].time_stamp)
				target = i;
	
	task_list.emplace_back(new RunningTask(target, clock()));
}

void StrategyManager::getSchedule(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) {
	p_strategy->schedulingAlgorithm(query_information_list, task_list);
}

StrategyManager::~StrategyManager() {
	delete p_strategy;
}