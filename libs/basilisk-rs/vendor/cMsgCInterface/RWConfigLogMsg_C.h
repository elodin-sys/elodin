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

#ifndef RWConfigLogMsg_C_H
#define RWConfigLogMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWConfigLogMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWConfigLogMsgPayload payload;		        //!< message copy, zero'd on construction
    RWConfigLogMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWConfigLogMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWConfigLogMsg_cpp_subscribe(RWConfigLogMsg_C *subscriber, void* source);

void RWConfigLogMsg_C_subscribe(RWConfigLogMsg_C *subscriber, RWConfigLogMsg_C *source);

int8_t RWConfigLogMsg_C_isSubscribedTo(RWConfigLogMsg_C *subscriber, RWConfigLogMsg_C *source);
int8_t RWConfigLogMsg_cpp_isSubscribedTo(RWConfigLogMsg_C *subscriber, void* source);

void RWConfigLogMsg_C_addAuthor(RWConfigLogMsg_C *coowner, RWConfigLogMsg_C *data);

void RWConfigLogMsg_C_init(RWConfigLogMsg_C *owner);

int RWConfigLogMsg_C_isLinked(RWConfigLogMsg_C *data);

int RWConfigLogMsg_C_isWritten(RWConfigLogMsg_C *data);

uint64_t RWConfigLogMsg_C_timeWritten(RWConfigLogMsg_C *data);

int64_t RWConfigLogMsg_C_moduleID(RWConfigLogMsg_C *data);

void RWConfigLogMsg_C_write(RWConfigLogMsgPayload *data, RWConfigLogMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWConfigLogMsgPayload RWConfigLogMsg_C_read(RWConfigLogMsg_C *source);

RWConfigLogMsgPayload RWConfigLogMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif