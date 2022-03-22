#include <iostream>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <thread>
#include <vector>
#include <carlsim.h>

#include "Message.h"
#include "MessageQueue.h"
#include "Microbrain.h"

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
