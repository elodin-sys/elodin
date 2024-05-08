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

#ifndef RWCmdMsg_C_H
#define RWCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    RWCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWCmdMsg_cpp_subscribe(RWCmdMsg_C *subscriber, void* source);

void RWCmdMsg_C_subscribe(RWCmdMsg_C *subscriber, RWCmdMsg_C *source);

int8_t RWCmdMsg_C_isSubscribedTo(RWCmdMsg_C *subscriber, RWCmdMsg_C *source);
int8_t RWCmdMsg_cpp_isSubscribedTo(RWCmdMsg_C *subscriber, void* source);

void RWCmdMsg_C_addAuthor(RWCmdMsg_C *coowner, RWCmdMsg_C *data);

void RWCmdMsg_C_init(RWCmdMsg_C *owner);

int RWCmdMsg_C_isLinked(RWCmdMsg_C *data);

int RWCmdMsg_C_isWritten(RWCmdMsg_C *data);

uint64_t RWCmdMsg_C_timeWritten(RWCmdMsg_C *data);

int64_t RWCmdMsg_C_moduleID(RWCmdMsg_C *data);

void RWCmdMsg_C_write(RWCmdMsgPayload *data, RWCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWCmdMsgPayload RWCmdMsg_C_read(RWCmdMsg_C *source);

RWCmdMsgPayload RWCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif