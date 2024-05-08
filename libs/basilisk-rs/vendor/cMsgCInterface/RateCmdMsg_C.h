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

#ifndef RateCmdMsg_C_H
#define RateCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RateCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RateCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    RateCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RateCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RateCmdMsg_cpp_subscribe(RateCmdMsg_C *subscriber, void* source);

void RateCmdMsg_C_subscribe(RateCmdMsg_C *subscriber, RateCmdMsg_C *source);

int8_t RateCmdMsg_C_isSubscribedTo(RateCmdMsg_C *subscriber, RateCmdMsg_C *source);
int8_t RateCmdMsg_cpp_isSubscribedTo(RateCmdMsg_C *subscriber, void* source);

void RateCmdMsg_C_addAuthor(RateCmdMsg_C *coowner, RateCmdMsg_C *data);

void RateCmdMsg_C_init(RateCmdMsg_C *owner);

int RateCmdMsg_C_isLinked(RateCmdMsg_C *data);

int RateCmdMsg_C_isWritten(RateCmdMsg_C *data);

uint64_t RateCmdMsg_C_timeWritten(RateCmdMsg_C *data);

int64_t RateCmdMsg_C_moduleID(RateCmdMsg_C *data);

void RateCmdMsg_C_write(RateCmdMsgPayload *data, RateCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

RateCmdMsgPayload RateCmdMsg_C_read(RateCmdMsg_C *source);

RateCmdMsgPayload RateCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif