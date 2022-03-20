#include <chrono>
#include <condition_variable>
#include <queue>
#include <mutex>
#include <map>
#include <utility>

#include "Message.h"
#include "MessageQueue.h"

MessageQueue::MessageQueue(): msg_que(), que_lock(), que_condition(), response_map(), map_lock() {}
MessageQueue::~MessageQueue() {}

void MessageQueue::put(Message&& msg) {
	// lock and push
	{
		std::lock_guard<std::mutex> lock(que_lock);
		msg_que.push(msg.move());
	}
	// notify one
	que_condition.notify_one();
}

//todo: maybe a timeout here
std::unique_ptr<Message> MessageQueue::get() {
	// wait for the lock
	std::unique_lock<std::mutex> lock(que_lock);
	que_condition.wait(lock, [this]{return !msg_que.empty();});

	// return the front of queue
	auto ret_msg = msg_que.front()->move();
	msg_que.pop();
	return ret_msg;
}

//std::unique_ptr<Message> request(Message&& msg) {}
//std::bool respondTo(MsgUID req_uid, Message&& response_msg) {}