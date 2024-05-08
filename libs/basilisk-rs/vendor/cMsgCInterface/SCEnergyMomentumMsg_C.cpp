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

/* All of the files in this folder (dist3/autoSource) are autocoded by the script
architecture/messaging/msgAutoSource/GenCMessages.py.
The script checks for the line "INSTANTIATE_TEMPLATES" in the file architecture/messaging/messaging.i. This
ensures that if a c++ message is instantiated that we also have a C equivalent of that message.
*/

#include "SCEnergyMomentumMsg_C.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/bsk_Print.h"
#include<string.h>

//! C interface to subscribe to a message
void SCEnergyMomentumMsg_C_subscribe(SCEnergyMomentumMsg_C *subscriber, SCEnergyMomentumMsg_C *source) {
    subscriber->payloadPointer = &(source->payload);
    subscriber->headerPointer = &(source->header);
    subscriber->header.isLinked = 1;            // set input message as linked
    subscriber->headerPointer->isLinked = 1;        // set output message as linked
};

//! C interface to check if subscriber is indeed subscribed to a message (1: subscribed, 0: not subscribed)
int8_t SCEnergyMomentumMsg_C_isSubscribedTo(SCEnergyMomentumMsg_C *subscriber, SCEnergyMomentumMsg_C *source) {

    return ((subscriber->payloadPointer == &(source->payload))&&(subscriber->headerPointer == &(source->header)));
    
};

//! C interface to claim authorship to a message
void SCEnergyMomentumMsg_C_addAuthor(SCEnergyMomentumMsg_C *coownerMsg, SCEnergyMomentumMsg_C *targetMsg) {
    coownerMsg->payloadPointer = &(targetMsg->payload);
    coownerMsg->headerPointer = &(targetMsg->header);
};

//! C interface to initialize the module output message
void SCEnergyMomentumMsg_C_init(SCEnergyMomentumMsg_C *owner) {
    //! check if the msg pointer is not assigned already.  If not, then connect message to itself.
    if (owner->payloadPointer == 0) {
        SCEnergyMomentumMsg_C_addAuthor(owner, owner);
    }
};

//! C interface to write to a message
void SCEnergyMomentumMsg_C_write(SCEnergyMomentumMsgPayload *data, SCEnergyMomentumMsg_C *destination, int64_t moduleID, uint64_t callTime) {
    *destination->payloadPointer = *data;
    destination->headerPointer->isWritten = 1;
    destination->headerPointer->timeWritten = callTime;
    destination->headerPointer->moduleID = moduleID;
    return;
};

//! C interface to return a zero'd copy of the message payload
SCEnergyMomentumMsgPayload SCEnergyMomentumMsg_C_zeroMsgPayload() {
    SCEnergyMomentumMsgPayload zeroMsg;
    memset(&zeroMsg, 0x0, sizeof(SCEnergyMomentumMsgPayload));
    return zeroMsg;
};


//! C interface to read to a message
SCEnergyMomentumMsgPayload SCEnergyMomentumMsg_C_read(SCEnergyMomentumMsg_C *source) {
    if (!source->headerPointer->isWritten) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are trying to read an un-written message of type SCEnergyMomentumMsg.");
    }
    //! ensure the current message container has a copy of a subscribed message.
    //! Does nothing if the message is writing to itself
    source->payload = *source->payloadPointer;

    return *source->payloadPointer;
};

//! C interface to see if this message container has been subscribed to
int SCEnergyMomentumMsg_C_isLinked(SCEnergyMomentumMsg_C *data) {
    return (int) data->header.isLinked;
};

//! C interface to see if this message container ever been written to
int SCEnergyMomentumMsg_C_isWritten(SCEnergyMomentumMsg_C *data) {
    if (data->header.isLinked) {
        return (int) data->headerPointer->isWritten;
    }
    BSK_PRINT(MSG_ERROR,"In C input msg, you are checking if an unconnected msg of type SCEnergyMomentumMsg is written.");
    return 0;
};

//! C interface to see if this message container ever been written to
uint64_t SCEnergyMomentumMsg_C_timeWritten(SCEnergyMomentumMsg_C *data) {
    if (data->header.isLinked) {
        return data->headerPointer->timeWritten;
    }
    BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting the write time of an unconnected msg of type SCEnergyMomentumMsg.");
    return 0;
};

//! C interface to get the moduleID of who wrote the message
int64_t SCEnergyMomentumMsg_C_moduleID(SCEnergyMomentumMsg_C *data) {
    if (!data->header.isLinked) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting moduleID of an unconnected msg of type SCEnergyMomentumMsg.");
        return 0;
    }
    if (!data->headerPointer->isWritten) {
        BSK_PRINT(MSG_ERROR,"In C input msg, you are requesting moduleID of an unwritten msg of type SCEnergyMomentumMsg.");
        return 0;
    }
    return data->headerPointer->moduleID;
};

//! method description
void SCEnergyMomentumMsg_cpp_subscribe(SCEnergyMomentumMsg_C *subscriber, void* source){
    Message<SCEnergyMomentumMsgPayload>* source_t = (Message<SCEnergyMomentumMsgPayload>*) source;
    MsgHeader *msgPtr;
    subscriber->payloadPointer = source_t->subscribeRaw(&(msgPtr));
    subscriber->headerPointer = msgPtr;
    subscriber->header.isLinked = 1;    // set input message as linked
    subscriber->headerPointer->isLinked = 1;    // set output message as linked
};


//! Cpp interface to check if subscriber is indeed subscribed to a message (1: subscribed, 0: not subscribed)
int8_t SCEnergyMomentumMsg_cpp_isSubscribedTo(SCEnergyMomentumMsg_C *subscriber, void* source) {

    MsgHeader *dummyMsgPtr;
    Message<SCEnergyMomentumMsgPayload>* source_t = (Message<SCEnergyMomentumMsgPayload>*) source;
    int8_t firstCheck = (subscriber->payloadPointer == source_t->getMsgPointers(&(dummyMsgPtr)));
    int8_t secondCheck = (subscriber->headerPointer == dummyMsgPtr);

    return (firstCheck && secondCheck);
    
};
