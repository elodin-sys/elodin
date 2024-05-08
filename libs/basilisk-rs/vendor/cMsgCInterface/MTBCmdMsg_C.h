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

#ifndef MTBCmdMsg_C_H
#define MTBCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/MTBCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    MTBCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    MTBCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} MTBCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void MTBCmdMsg_cpp_subscribe(MTBCmdMsg_C *subscriber, void* source);

void MTBCmdMsg_C_subscribe(MTBCmdMsg_C *subscriber, MTBCmdMsg_C *source);

int8_t MTBCmdMsg_C_isSubscribedTo(MTBCmdMsg_C *subscriber, MTBCmdMsg_C *source);
int8_t MTBCmdMsg_cpp_isSubscribedTo(MTBCmdMsg_C *subscriber, void* source);

void MTBCmdMsg_C_addAuthor(MTBCmdMsg_C *coowner, MTBCmdMsg_C *data);

void MTBCmdMsg_C_init(MTBCmdMsg_C *owner);

int MTBCmdMsg_C_isLinked(MTBCmdMsg_C *data);

int MTBCmdMsg_C_isWritten(MTBCmdMsg_C *data);

uint64_t MTBCmdMsg_C_timeWritten(MTBCmdMsg_C *data);

int64_t MTBCmdMsg_C_moduleID(MTBCmdMsg_C *data);

void MTBCmdMsg_C_write(MTBCmdMsgPayload *data, MTBCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

MTBCmdMsgPayload MTBCmdMsg_C_read(MTBCmdMsg_C *source);

MTBCmdMsgPayload MTBCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif