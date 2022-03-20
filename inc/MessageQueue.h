#ifndef _MESSAGEQUEUE_H
#define _MESSAGEQUEUE_H

class MessageQueue {
	public:
		MessageQueue();
		~MessageQueue();

		// put/get message to the queue
		void put(Message&& msg);
		std::unique_ptr<Message> get();

		//request/respond to the message
		//std::unique_ptr<Message> request(Message&& msg);
		//bool respondTo(MsgUID req_uid, Message&& response_msg);

		struct Request {
			Request () {};

			std::unique_ptr<Message> response;					// response message
			std::condition_variable condition_var;			// condition var
		};

	private:
		// message queue with lock
		std::queue<std::unique_ptr<Message>> msg_que;
		std::mutex que_lock;
		std::condition_variable que_condition;

		// response map
		std::map<MsgUID, Request*> response_map;
		std::mutex map_lock;
};

#endif