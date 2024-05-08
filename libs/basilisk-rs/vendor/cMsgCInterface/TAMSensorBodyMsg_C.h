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

#ifndef TAMSensorBodyMsg_C_H
#define TAMSensorBodyMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/TAMSensorBodyMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    TAMSensorBodyMsgPayload payload;		        //!< message copy, zero'd on construction
    TAMSensorBodyMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} TAMSensorBodyMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void TAMSensorBodyMsg_cpp_subscribe(TAMSensorBodyMsg_C *subscriber, void* source);

void TAMSensorBodyMsg_C_subscribe(TAMSensorBodyMsg_C *subscriber, TAMSensorBodyMsg_C *source);

int8_t TAMSensorBodyMsg_C_isSubscribedTo(TAMSensorBodyMsg_C *subscriber, TAMSensorBodyMsg_C *source);
int8_t TAMSensorBodyMsg_cpp_isSubscribedTo(TAMSensorBodyMsg_C *subscriber, void* source);

void TAMSensorBodyMsg_C_addAuthor(TAMSensorBodyMsg_C *coowner, TAMSensorBodyMsg_C *data);

void TAMSensorBodyMsg_C_init(TAMSensorBodyMsg_C *owner);

int TAMSensorBodyMsg_C_isLinked(TAMSensorBodyMsg_C *data);

int TAMSensorBodyMsg_C_isWritten(TAMSensorBodyMsg_C *data);

uint64_t TAMSensorBodyMsg_C_timeWritten(TAMSensorBodyMsg_C *data);

int64_t TAMSensorBodyMsg_C_moduleID(TAMSensorBodyMsg_C *data);

void TAMSensorBodyMsg_C_write(TAMSensorBodyMsgPayload *data, TAMSensorBodyMsg_C *destination, int64_t moduleID, uint64_t callTime);

TAMSensorBodyMsgPayload TAMSensorBodyMsg_C_read(TAMSensorBodyMsg_C *source);

TAMSensorBodyMsgPayload TAMSensorBodyMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif