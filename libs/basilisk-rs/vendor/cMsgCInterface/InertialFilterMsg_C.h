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

#ifndef InertialFilterMsg_C_H
#define InertialFilterMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/InertialFilterMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    InertialFilterMsgPayload payload;		        //!< message copy, zero'd on construction
    InertialFilterMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} InertialFilterMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void InertialFilterMsg_cpp_subscribe(InertialFilterMsg_C *subscriber, void* source);

void InertialFilterMsg_C_subscribe(InertialFilterMsg_C *subscriber, InertialFilterMsg_C *source);

int8_t InertialFilterMsg_C_isSubscribedTo(InertialFilterMsg_C *subscriber, InertialFilterMsg_C *source);
int8_t InertialFilterMsg_cpp_isSubscribedTo(InertialFilterMsg_C *subscriber, void* source);

void InertialFilterMsg_C_addAuthor(InertialFilterMsg_C *coowner, InertialFilterMsg_C *data);

void InertialFilterMsg_C_init(InertialFilterMsg_C *owner);

int InertialFilterMsg_C_isLinked(InertialFilterMsg_C *data);

int InertialFilterMsg_C_isWritten(InertialFilterMsg_C *data);

uint64_t InertialFilterMsg_C_timeWritten(InertialFilterMsg_C *data);

int64_t InertialFilterMsg_C_moduleID(InertialFilterMsg_C *data);

void InertialFilterMsg_C_write(InertialFilterMsgPayload *data, InertialFilterMsg_C *destination, int64_t moduleID, uint64_t callTime);

InertialFilterMsgPayload InertialFilterMsg_C_read(InertialFilterMsg_C *source);

InertialFilterMsgPayload InertialFilterMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif