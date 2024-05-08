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

#ifndef TAMSensorMsg_C_H
#define TAMSensorMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/TAMSensorMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    TAMSensorMsgPayload payload;		        //!< message copy, zero'd on construction
    TAMSensorMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} TAMSensorMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void TAMSensorMsg_cpp_subscribe(TAMSensorMsg_C *subscriber, void* source);

void TAMSensorMsg_C_subscribe(TAMSensorMsg_C *subscriber, TAMSensorMsg_C *source);

int8_t TAMSensorMsg_C_isSubscribedTo(TAMSensorMsg_C *subscriber, TAMSensorMsg_C *source);
int8_t TAMSensorMsg_cpp_isSubscribedTo(TAMSensorMsg_C *subscriber, void* source);

void TAMSensorMsg_C_addAuthor(TAMSensorMsg_C *coowner, TAMSensorMsg_C *data);

void TAMSensorMsg_C_init(TAMSensorMsg_C *owner);

int TAMSensorMsg_C_isLinked(TAMSensorMsg_C *data);

int TAMSensorMsg_C_isWritten(TAMSensorMsg_C *data);

uint64_t TAMSensorMsg_C_timeWritten(TAMSensorMsg_C *data);

int64_t TAMSensorMsg_C_moduleID(TAMSensorMsg_C *data);

void TAMSensorMsg_C_write(TAMSensorMsgPayload *data, TAMSensorMsg_C *destination, int64_t moduleID, uint64_t callTime);

TAMSensorMsgPayload TAMSensorMsg_C_read(TAMSensorMsg_C *source);

TAMSensorMsgPayload TAMSensorMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif