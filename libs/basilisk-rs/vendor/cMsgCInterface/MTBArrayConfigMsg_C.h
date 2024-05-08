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

#ifndef MTBArrayConfigMsg_C_H
#define MTBArrayConfigMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/MTBArrayConfigMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    MTBArrayConfigMsgPayload payload;		        //!< message copy, zero'd on construction
    MTBArrayConfigMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} MTBArrayConfigMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void MTBArrayConfigMsg_cpp_subscribe(MTBArrayConfigMsg_C *subscriber, void* source);

void MTBArrayConfigMsg_C_subscribe(MTBArrayConfigMsg_C *subscriber, MTBArrayConfigMsg_C *source);

int8_t MTBArrayConfigMsg_C_isSubscribedTo(MTBArrayConfigMsg_C *subscriber, MTBArrayConfigMsg_C *source);
int8_t MTBArrayConfigMsg_cpp_isSubscribedTo(MTBArrayConfigMsg_C *subscriber, void* source);

void MTBArrayConfigMsg_C_addAuthor(MTBArrayConfigMsg_C *coowner, MTBArrayConfigMsg_C *data);

void MTBArrayConfigMsg_C_init(MTBArrayConfigMsg_C *owner);

int MTBArrayConfigMsg_C_isLinked(MTBArrayConfigMsg_C *data);

int MTBArrayConfigMsg_C_isWritten(MTBArrayConfigMsg_C *data);

uint64_t MTBArrayConfigMsg_C_timeWritten(MTBArrayConfigMsg_C *data);

int64_t MTBArrayConfigMsg_C_moduleID(MTBArrayConfigMsg_C *data);

void MTBArrayConfigMsg_C_write(MTBArrayConfigMsgPayload *data, MTBArrayConfigMsg_C *destination, int64_t moduleID, uint64_t callTime);

MTBArrayConfigMsgPayload MTBArrayConfigMsg_C_read(MTBArrayConfigMsg_C *source);

MTBArrayConfigMsgPayload MTBArrayConfigMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif