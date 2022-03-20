#ifndef _MESSAGE_H
#define _MESSAGE_H

using MsgUID = unsigned long long;

class Message {
	public:
		Message (int message_id);
		virtual ~Message() = default;
    	Message& operator=(const Message&) = delete;

		// for override
		virtual std::unique_ptr<Message> move();

		// get value
		int getMessageId();
		MsgUID getUniqueId();
	protected:
		Message(Message&&) = default;
		Message& operator=(Message&&) = default;

	private:
		int message_id;
		MsgUID unique_id;
};


template <typename PayloadType>
class DataMessage : public Message {
	public:
		template <typename ... Args> DataMessage(int message_id, Args&& ... args):
			Message(message_id), 
			payload(new PayloadType(std::forward<Args>(args) ...))
			{}

		virtual ~DataMessage() = default;
		DataMessage(const DataMessage&) = delete;
		DataMessage& operator=(const DataMessage&) = delete;

		virtual std::unique_ptr<Message> move() override {
        	return std::unique_ptr<Message>(new DataMessage<PayloadType>(std::move(*this)));
		}
		PayloadType& getPayload() {
			return *payload;
		}
	protected:
		DataMessage(DataMessage&&) = default;
		DataMessage& operator=(DataMessage&&) = default;
	private:
		std::unique_ptr<PayloadType> payload;
};

#endif