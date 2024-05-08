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

#ifndef GroundStateMsg_C_H
#define GroundStateMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/GroundStateMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    GroundStateMsgPayload payload;		        //!< message copy, zero'd on construction
    GroundStateMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} GroundStateMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void GroundStateMsg_cpp_subscribe(GroundStateMsg_C *subscriber, void* source);

void GroundStateMsg_C_subscribe(GroundStateMsg_C *subscriber, GroundStateMsg_C *source);

int8_t GroundStateMsg_C_isSubscribedTo(GroundStateMsg_C *subscriber, GroundStateMsg_C *source);
int8_t GroundStateMsg_cpp_isSubscribedTo(GroundStateMsg_C *subscriber, void* source);

void GroundStateMsg_C_addAuthor(GroundStateMsg_C *coowner, GroundStateMsg_C *data);

void GroundStateMsg_C_init(GroundStateMsg_C *owner);

int GroundStateMsg_C_isLinked(GroundStateMsg_C *data);

int GroundStateMsg_C_isWritten(GroundStateMsg_C *data);

uint64_t GroundStateMsg_C_timeWritten(GroundStateMsg_C *data);

int64_t GroundStateMsg_C_moduleID(GroundStateMsg_C *data);

void GroundStateMsg_C_write(GroundStateMsgPayload *data, GroundStateMsg_C *destination, int64_t moduleID, uint64_t callTime);

GroundStateMsgPayload GroundStateMsg_C_read(GroundStateMsg_C *source);

GroundStateMsgPayload GroundStateMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif