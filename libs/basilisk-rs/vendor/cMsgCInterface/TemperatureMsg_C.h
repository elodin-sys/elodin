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

#ifndef TemperatureMsg_C_H
#define TemperatureMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/TemperatureMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    TemperatureMsgPayload payload;		        //!< message copy, zero'd on construction
    TemperatureMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} TemperatureMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void TemperatureMsg_cpp_subscribe(TemperatureMsg_C *subscriber, void* source);

void TemperatureMsg_C_subscribe(TemperatureMsg_C *subscriber, TemperatureMsg_C *source);

int8_t TemperatureMsg_C_isSubscribedTo(TemperatureMsg_C *subscriber, TemperatureMsg_C *source);
int8_t TemperatureMsg_cpp_isSubscribedTo(TemperatureMsg_C *subscriber, void* source);

void TemperatureMsg_C_addAuthor(TemperatureMsg_C *coowner, TemperatureMsg_C *data);

void TemperatureMsg_C_init(TemperatureMsg_C *owner);

int TemperatureMsg_C_isLinked(TemperatureMsg_C *data);

int TemperatureMsg_C_isWritten(TemperatureMsg_C *data);

uint64_t TemperatureMsg_C_timeWritten(TemperatureMsg_C *data);

int64_t TemperatureMsg_C_moduleID(TemperatureMsg_C *data);

void TemperatureMsg_C_write(TemperatureMsgPayload *data, TemperatureMsg_C *destination, int64_t moduleID, uint64_t callTime);

TemperatureMsgPayload TemperatureMsg_C_read(TemperatureMsg_C *source);

TemperatureMsgPayload TemperatureMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif