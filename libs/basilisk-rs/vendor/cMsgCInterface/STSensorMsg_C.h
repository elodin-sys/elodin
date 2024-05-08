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

#ifndef STSensorMsg_C_H
#define STSensorMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/STSensorMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    STSensorMsgPayload payload;		        //!< message copy, zero'd on construction
    STSensorMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} STSensorMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void STSensorMsg_cpp_subscribe(STSensorMsg_C *subscriber, void* source);

void STSensorMsg_C_subscribe(STSensorMsg_C *subscriber, STSensorMsg_C *source);

int8_t STSensorMsg_C_isSubscribedTo(STSensorMsg_C *subscriber, STSensorMsg_C *source);
int8_t STSensorMsg_cpp_isSubscribedTo(STSensorMsg_C *subscriber, void* source);

void STSensorMsg_C_addAuthor(STSensorMsg_C *coowner, STSensorMsg_C *data);

void STSensorMsg_C_init(STSensorMsg_C *owner);

int STSensorMsg_C_isLinked(STSensorMsg_C *data);

int STSensorMsg_C_isWritten(STSensorMsg_C *data);

uint64_t STSensorMsg_C_timeWritten(STSensorMsg_C *data);

int64_t STSensorMsg_C_moduleID(STSensorMsg_C *data);

void STSensorMsg_C_write(STSensorMsgPayload *data, STSensorMsg_C *destination, int64_t moduleID, uint64_t callTime);

STSensorMsgPayload STSensorMsg_C_read(STSensorMsg_C *source);

STSensorMsgPayload STSensorMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif