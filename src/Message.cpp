#include <memory>
#include <utility>
#include <atomic>

#include "Message.h"

MsgUID generate_unique_id() {
	static std::atomic<MsgUID> id(0);
	return ++id;
}

Message::Message(int msg_id) : message_id(msg_id), unique_id(generate_unique_id()) {} 

std::unique_ptr<Message> Message::move() {
	return std::unique_ptr<Message>(new Message(std::move(*this)));
}

int Message::getMessageId() {
	return message_id;
}

MsgUID Message::getUniqueId() {
	return unique_id;
}

/*
template <typename PayloadType>
std::unique_ptr<Message> DataMessage<PayloadType>::move() {
	return std::unique_ptr<Message>(new DataMessage<PayloadType>(std::move(*this)));
}

template <typename PayloadType>
PayloadType& DataMessage<PayloadType>::getPayload() {
	return *payload;
}
*/