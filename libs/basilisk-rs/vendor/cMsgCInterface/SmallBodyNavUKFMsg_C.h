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

#ifndef SmallBodyNavUKFMsg_C_H
#define SmallBodyNavUKFMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SmallBodyNavUKFMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SmallBodyNavUKFMsgPayload payload;		        //!< message copy, zero'd on construction
    SmallBodyNavUKFMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SmallBodyNavUKFMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SmallBodyNavUKFMsg_cpp_subscribe(SmallBodyNavUKFMsg_C *subscriber, void* source);

void SmallBodyNavUKFMsg_C_subscribe(SmallBodyNavUKFMsg_C *subscriber, SmallBodyNavUKFMsg_C *source);

int8_t SmallBodyNavUKFMsg_C_isSubscribedTo(SmallBodyNavUKFMsg_C *subscriber, SmallBodyNavUKFMsg_C *source);
int8_t SmallBodyNavUKFMsg_cpp_isSubscribedTo(SmallBodyNavUKFMsg_C *subscriber, void* source);

void SmallBodyNavUKFMsg_C_addAuthor(SmallBodyNavUKFMsg_C *coowner, SmallBodyNavUKFMsg_C *data);

void SmallBodyNavUKFMsg_C_init(SmallBodyNavUKFMsg_C *owner);

int SmallBodyNavUKFMsg_C_isLinked(SmallBodyNavUKFMsg_C *data);

int SmallBodyNavUKFMsg_C_isWritten(SmallBodyNavUKFMsg_C *data);

uint64_t SmallBodyNavUKFMsg_C_timeWritten(SmallBodyNavUKFMsg_C *data);

int64_t SmallBodyNavUKFMsg_C_moduleID(SmallBodyNavUKFMsg_C *data);

void SmallBodyNavUKFMsg_C_write(SmallBodyNavUKFMsgPayload *data, SmallBodyNavUKFMsg_C *destination, int64_t moduleID, uint64_t callTime);

SmallBodyNavUKFMsgPayload SmallBodyNavUKFMsg_C_read(SmallBodyNavUKFMsg_C *source);

SmallBodyNavUKFMsgPayload SmallBodyNavUKFMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif