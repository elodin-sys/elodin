/*
ISC License

Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

Permission to use, copy, modify, and/or distribute this software for any
        purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
        ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
        OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
*/
#ifndef MESSAGING_H
#define MESSAGING_H
#include <memory>
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include <vector>
#include "architecture/messaging/msgHeader.h"
#include "architecture/utilities/bskLogging.h"
#include <typeinfo>
#include <stdlib.h>

/*! forward-declare sim message for use by read functor */
template<typename messageType>
class Message;

template<typename messageType>
class Recorder;

/*! Read functors have read-only access to messages*/
template<typename messageType>
class ReadFunctor{
private:
    messageType* payloadPointer;    //!< -- pointer to the incoming msg data
    MsgHeader *headerPointer;      //!< -- pointer to the incoming msg header
    bool initialized;               //!< -- flag indicating if the input message is connect to another message
    
public:
    //!< -- BSK Logging
    BSKLogger bskLogger;            //!< -- bsk logging instance
    messageType zeroMsgPayload ={}; //!< -- zero'd copy of the message payload type


    //! constructor
    ReadFunctor() : initialized(false) {};

    //! constructor
    ReadFunctor(messageType* payloadPtr, MsgHeader *headerPtr) : payloadPointer(payloadPtr), headerPointer(headerPtr), initialized(true){};

    //! constructor
    const messageType& operator()(){
        if (!this->initialized) {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are trying to read an un-connected message of type %s\nThis program is about to self destruct.",  typeid(var).name());
        }
        return *this->payloadPointer;

    };

    //! check if this msg has been connected to
    bool isLinked(){return this->initialized;};  // something that can be checked so that uninitialized messages aren't read.

    //! check if the message has been ever written to
    bool isWritten(){
        if (this->initialized) {
            return this->headerPointer->isWritten;
        } else {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are checking if an unconnected msg of type %s is written.", typeid(var).name());
            return false;
        }
    };

    //! return the time at which the message was written
    uint64_t timeWritten(){
        if (!this->initialized) {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are requesting the write time of an unconnected msg of type %s.", typeid(var).name());
            return 0;
        }
        return this->headerPointer->timeWritten;
    };

    //! return the moduleID of who wrote wrote the message
    int64_t moduleID(){
        if (!this->initialized) {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are requesting moduleID of an unconnected msg of type %s.", typeid(var).name());
            return 0;
        }
        if (!this->headerPointer->isWritten) {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are requesting moduleID of an unwritten msg of type %s.", typeid(var).name());
            return 0;
        }
        return this->headerPointer->moduleID;
    };

    //! subscribe to a C message
    void subscribeToC(void* source){
        // this method works by knowing that the first member of a C message is the header.
        this->headerPointer = (MsgHeader*) source;

        // advance the address to connect to C-wrapped message payload
        // this assumes the header memory is aligned with 0 additional padding
        MsgHeader* pt = this->headerPointer;
        this->payloadPointer = (messageType *) (++pt);


        // set flag that this input message is connected to another message
        this->initialized = true;           // set input message as linked
        this->headerPointer->isLinked = 1;  // set source output message as linked
    };

    //! Subscribe to a C++ message
    void subscribeTo(Message<messageType> *source){
        *this = source->addSubscriber();
        this->initialized = true;
    };


    //! Check if self has been subscribed to a C message
    uint8_t isSubscribedToC(void *source){
        
        int8_t firstCheck = (this->headerPointer == (MsgHeader*) source);
        MsgHeader* pt = this->headerPointer;
        int8_t secondCheck = (this->payloadPointer == (messageType *) (++pt));

        return (this->initialized && firstCheck && secondCheck);
        
    };
    //! Check if self has been subscribed to a Cpp message
    uint8_t isSubscribedTo(Message<messageType> *source){
        
        MsgHeader *dummyMsgPtr;
        int8_t firstCheck = (this->payloadPointer == source->getMsgPointers(&(dummyMsgPtr)));
        int8_t secondCheck = (this->headerPointer == dummyMsgPtr);

        return (this->initialized && firstCheck && secondCheck );

    };

    //! Recorder method description
    Recorder<messageType> recorder(uint64_t timeDiff = 0){return Recorder<messageType>(this, timeDiff);}
};

/*! Write Functor */
template<typename messageType>
class WriteFunctor{
private:
    messageType* payloadPointer;    //!< pointer to the message payload
    MsgHeader* headerPointer;       //!< pointer to the message header
public:
    //! write functor constructor
    WriteFunctor(){};
    //! write functor constructor
    WriteFunctor(messageType* payloadPointer, MsgHeader *headerPointer) : payloadPointer(payloadPointer), headerPointer(headerPointer){};
    //! write functor constructor
    void operator()(messageType *payload, int64_t moduleID, uint64_t callTime){
        *this->payloadPointer = *payload;
        this->headerPointer->isWritten = 1;
        this->headerPointer->timeWritten = callTime;
        this->headerPointer->moduleID = moduleID;
        return;
    }
};

template<typename messageType>
class Recorder;

/*!
 * base class template for bsk messages
 */
template<typename messageType>
class Message{
private:
    messageType payload = {};   //!< struct defining message payload, zero'd on creation
    MsgHeader header = {};      //!< struct defining the message header, zero'd on creation
    ReadFunctor<messageType> read = ReadFunctor<messageType>(&payload, &header);  //!< read functor instance
public:
    //! write functor to this message
    WriteFunctor<messageType> write = WriteFunctor<messageType>(&payload, &header);
    //! -- request read rights. returns reference to class ``read`` variable
    ReadFunctor<messageType> addSubscriber();
    //! -- request write rights.
    WriteFunctor<messageType> addAuthor();
    //! for plain ole c modules
    messageType* subscribeRaw(MsgHeader **msgPtr);

    //! for plain ole c modules
    messageType* getMsgPointers(MsgHeader **msgPtr);

    //! Recorder object
    Recorder<messageType> recorder(uint64_t timeDiff = 0){return Recorder<messageType>(this, timeDiff);}
    
    messageType zeroMsgPayload = {};    //!< zero'd copy of the message payload structure

    //! check if this msg has been connected to
    bool isLinked(){return this->header.isLinked;};

    //! Return the memory size of the payload, be careful about dynamically sized things
    uint64_t getPayloadSize() {return sizeof(messageType);};
};


template<typename messageType>
ReadFunctor<messageType> Message<messageType>::addSubscriber(){
    this->header.isLinked = 1;
    return this->read;
}

template<typename messageType>
WriteFunctor<messageType> Message<messageType>::addAuthor(){
    return this->write;
}

template<typename messageType>
messageType* Message<messageType>::subscribeRaw(MsgHeader **msgPtr){
    *msgPtr = &this->header;
    this->header.isLinked = 1;
    return &this->payload;
}

template<typename messageType>
messageType* Message<messageType>::getMsgPointers(MsgHeader **msgPtr){
    *msgPtr = &this->header;
    return &this->payload;
}

/*! Keep a time history of messages accessible to users from python */
template<typename messageType>
class Recorder : public SysModel{
public:
    Recorder(){};
    //! -- Use this to record cpp messages
    Recorder(Message<messageType>* message, uint64_t timeDiff = 0){
        this->timeInterval = timeDiff;
        this->readMessage = message->addSubscriber();
        this->ModelTag = "Rec:" + findMsgName(std::string(typeid(*message).name()));
    }
    //! -- Use this to record C messages
    Recorder(void* message, uint64_t timeDiff = 0){
        this->timeInterval = timeDiff;

        MsgHeader* msgPt = (MsgHeader *) message;
        MsgHeader *pt = msgPt;
        messageType* payloadPointer;
        payloadPointer = (messageType *) (++pt);

        this->readMessage = ReadFunctor<messageType>(payloadPointer, msgPt);
        this->ModelTag = "Rec:";
        Message<messageType> tempMsg;
        std::string msgName = typeid(tempMsg).name();
        this->ModelTag += findMsgName(msgName);
    }
    //! -- Use this to keep track of what someone is reading
    Recorder(ReadFunctor<messageType>* messageReader, uint64_t timeDiff = 0){
        this->timeInterval = timeDiff;
        this->readMessage = *messageReader;
        if (!messageReader->isLinked()) {
            messageType var;
            bskLogger.bskLog(BSK_ERROR, "In C++ read functor, you are requesting to record an un-connected input message of type %s.", typeid(var).name());
        }
        this->ModelTag = "Rec:" + findMsgName(std::string(typeid(*messageReader).name()));
    }
    ~Recorder(){};

    //! -- self initialization
    void SelfInit(){};
    //! -- cross initialization
    void IntegratedInit(){};
    //! -- Read and record the message
    void UpdateState(uint64_t CurrentSimNanos){
        if (CurrentSimNanos >= this->nextUpdateTime) {
            this->msgRecordTimes.push_back(CurrentSimNanos);
            this->msgWrittenTimes.push_back(this->readMessage.timeWritten());
            this->msgRecord.push_back(this->readMessage());
            this->nextUpdateTime += this->timeInterval;
        }
    };
    //! Reset method
    void Reset(uint64_t CurrentSimNanos){
        this->msgRecord.clear();    //!< -- Can only reset to 0 for now
        this->msgRecordTimes.clear();
        this->msgWrittenTimes.clear();
        this->nextUpdateTime = CurrentSimNanos;
    };
    //! time recorded method
    std::vector<uint64_t>& times(){return this->msgRecordTimes;}
    //! time written method
    std::vector<uint64_t>& timesWritten(){return this->msgWrittenTimes;}
    //! record method
    std::vector<messageType>& record(){return this->msgRecord;};
    
    //! determine message name
    std::string findMsgName(std::string msgName) {
        size_t locMsg = msgName.find("Payload");
        if (locMsg != std::string::npos) {
            msgName.erase(locMsg, std::string::npos);
        }
        locMsg = msgName.find("Message");
        if (locMsg != std::string::npos) {
           msgName.replace(locMsg, 7, "");
        }
        for (int c = 0; c<10; c++) {
            locMsg = msgName.rfind(std::to_string(c));
            if (locMsg != std::string::npos) {
                msgName.erase(0, locMsg+1);
            }
        }
        return msgName;
    };

    //! clear the recorded messages, i.e. purge the history
    void clear(){
        this->msgRecord.clear();
        this->msgRecordTimes.clear();
        this->msgWrittenTimes.clear();
    };

    BSKLogger bskLogger;                          //!< -- BSK Logging

    //! method to update the minimum time interval before recording the next message
    void updateTimeInterval(uint64_t timeDiff) {
        this->timeInterval = timeDiff;
    };

private:
    std::vector<messageType> msgRecord;           //!< vector of recorded messages
    std::vector<uint64_t> msgRecordTimes;         //!< vector of times at which messages are recorded
    std::vector<uint64_t> msgWrittenTimes;        //!< vector of times at which messages are written
    uint64_t nextUpdateTime = 0;                  //!< [ns] earliest time at which the msg is recorded again
    uint64_t timeInterval;                        //!< [ns] recording time intervale

private:
    ReadFunctor<messageType> readMessage;   //!< method description
};

#endif
