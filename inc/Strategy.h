#ifndef _STRATEGY_H
#define _STRATEGY_H

#include "NetworkModel.h"

class StrategyInterface {
	public:
		virtual void schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) = 0;
};

class RoundRobinStrategy : public StrategyInterface {
	public:
		void schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) override;
};

class RandomStrategy : public StrategyInterface {
	public:
		void schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) override;
};

class WeightedRandomStrategy : public StrategyInterface {
	public:
		void schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) override;
};

class FCFSStrategy : public StrategyInterface {
	public:
		void schedulingAlgorithm(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list) override;
};

class StrategyManager {
	public:
		StrategyManager(StrategyInterface *p_strategy):p_strategy(p_strategy) {}

		void getSchedule(std::vector <QueryInformation> &query_information_list, std::vector <RunningTask> &task_list);

		~StrategyManager();

	private:
		StrategyInterface *p_strategy;
};

#endif