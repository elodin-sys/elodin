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

#ifndef RWArrayConfigMsg_C_H
#define RWArrayConfigMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWArrayConfigMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWArrayConfigMsgPayload payload;		        //!< message copy, zero'd on construction
    RWArrayConfigMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWArrayConfigMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWArrayConfigMsg_cpp_subscribe(RWArrayConfigMsg_C *subscriber, void* source);

void RWArrayConfigMsg_C_subscribe(RWArrayConfigMsg_C *subscriber, RWArrayConfigMsg_C *source);

int8_t RWArrayConfigMsg_C_isSubscribedTo(RWArrayConfigMsg_C *subscriber, RWArrayConfigMsg_C *source);
int8_t RWArrayConfigMsg_cpp_isSubscribedTo(RWArrayConfigMsg_C *subscriber, void* source);

void RWArrayConfigMsg_C_addAuthor(RWArrayConfigMsg_C *coowner, RWArrayConfigMsg_C *data);

void RWArrayConfigMsg_C_init(RWArrayConfigMsg_C *owner);

int RWArrayConfigMsg_C_isLinked(RWArrayConfigMsg_C *data);

int RWArrayConfigMsg_C_isWritten(RWArrayConfigMsg_C *data);

uint64_t RWArrayConfigMsg_C_timeWritten(RWArrayConfigMsg_C *data);

int64_t RWArrayConfigMsg_C_moduleID(RWArrayConfigMsg_C *data);

void RWArrayConfigMsg_C_write(RWArrayConfigMsgPayload *data, RWArrayConfigMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWArrayConfigMsgPayload RWArrayConfigMsg_C_read(RWArrayConfigMsg_C *source);

RWArrayConfigMsgPayload RWArrayConfigMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif