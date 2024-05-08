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

#ifndef RWSpeedMsg_C_H
#define RWSpeedMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWSpeedMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWSpeedMsgPayload payload;		        //!< message copy, zero'd on construction
    RWSpeedMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWSpeedMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWSpeedMsg_cpp_subscribe(RWSpeedMsg_C *subscriber, void* source);

void RWSpeedMsg_C_subscribe(RWSpeedMsg_C *subscriber, RWSpeedMsg_C *source);

int8_t RWSpeedMsg_C_isSubscribedTo(RWSpeedMsg_C *subscriber, RWSpeedMsg_C *source);
int8_t RWSpeedMsg_cpp_isSubscribedTo(RWSpeedMsg_C *subscriber, void* source);

void RWSpeedMsg_C_addAuthor(RWSpeedMsg_C *coowner, RWSpeedMsg_C *data);

void RWSpeedMsg_C_init(RWSpeedMsg_C *owner);

int RWSpeedMsg_C_isLinked(RWSpeedMsg_C *data);

int RWSpeedMsg_C_isWritten(RWSpeedMsg_C *data);

uint64_t RWSpeedMsg_C_timeWritten(RWSpeedMsg_C *data);

int64_t RWSpeedMsg_C_moduleID(RWSpeedMsg_C *data);

void RWSpeedMsg_C_write(RWSpeedMsgPayload *data, RWSpeedMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWSpeedMsgPayload RWSpeedMsg_C_read(RWSpeedMsg_C *source);

RWSpeedMsgPayload RWSpeedMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif