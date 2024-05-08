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

#ifndef SmallBodyNavMsg_C_H
#define SmallBodyNavMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SmallBodyNavMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNavMsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNavMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNavMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNavMsg_cpp_subscribe(SmallBodyNavMsg_C *subscriber, void* source);

void SmallBodyNavMsg_C_subscribe(SmallBodyNavMsg_C *subscriber, SmallBodyNavMsg_C *source);

int8_t SmallBodyNavMsg_C_isSubscribedTo(SmallBodyNavMsg_C *subscriber, SmallBodyNavMsg_C *source);
int8_t SmallBodyNavMsg_cpp_isSubscribedTo(SmallBodyNavMsg_C *subscriber, void* source);

void SmallBodyNavMsg_C_addAuthor(SmallBodyNavMsg_C *coowner, SmallBodyNavMsg_C *data);

void SmallBodyNavMsg_C_init(SmallBodyNavMsg_C *owner);

int SmallBodyNavMsg_C_isLinked(SmallBodyNavMsg_C *data);

int SmallBodyNavMsg_C_isWritten(SmallBodyNavMsg_C *data);

uint64_t SmallBodyNavMsg_C_timeWritten(SmallBodyNavMsg_C *data);

int64_t SmallBodyNavMsg_C_moduleID(SmallBodyNavMsg_C *data);

void SmallBodyNavMsg_C_write(SmallBodyNavMsgPayload *data, SmallBodyNavMsg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNavMsgPayload SmallBodyNavMsg_C_read(SmallBodyNavMsg_C *source);

SmallBodyNavMsgPayload SmallBodyNavMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif