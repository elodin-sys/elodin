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

#ifndef RWConstellationMsg_C_H
#define RWConstellationMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWConstellationMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWConstellationMsgPayload payload;		        //!< message copy, zero'd on construction
    RWConstellationMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWConstellationMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWConstellationMsg_cpp_subscribe(RWConstellationMsg_C *subscriber, void* source);

void RWConstellationMsg_C_subscribe(RWConstellationMsg_C *subscriber, RWConstellationMsg_C *source);

int8_t RWConstellationMsg_C_isSubscribedTo(RWConstellationMsg_C *subscriber, RWConstellationMsg_C *source);
int8_t RWConstellationMsg_cpp_isSubscribedTo(RWConstellationMsg_C *subscriber, void* source);

void RWConstellationMsg_C_addAuthor(RWConstellationMsg_C *coowner, RWConstellationMsg_C *data);

void RWConstellationMsg_C_init(RWConstellationMsg_C *owner);

int RWConstellationMsg_C_isLinked(RWConstellationMsg_C *data);

int RWConstellationMsg_C_isWritten(RWConstellationMsg_C *data);

uint64_t RWConstellationMsg_C_timeWritten(RWConstellationMsg_C *data);

int64_t RWConstellationMsg_C_moduleID(RWConstellationMsg_C *data);

void RWConstellationMsg_C_write(RWConstellationMsgPayload *data, RWConstellationMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWConstellationMsgPayload RWConstellationMsg_C_read(RWConstellationMsg_C *source);

RWConstellationMsgPayload RWConstellationMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif