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

#ifndef RWAvailabilityMsg_C_H
#define RWAvailabilityMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/RWAvailabilityMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    RWAvailabilityMsgPayload payload;		        //!< message copy, zero'd on construction
    RWAvailabilityMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} RWAvailabilityMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void RWAvailabilityMsg_cpp_subscribe(RWAvailabilityMsg_C *subscriber, void* source);

void RWAvailabilityMsg_C_subscribe(RWAvailabilityMsg_C *subscriber, RWAvailabilityMsg_C *source);

int8_t RWAvailabilityMsg_C_isSubscribedTo(RWAvailabilityMsg_C *subscriber, RWAvailabilityMsg_C *source);
int8_t RWAvailabilityMsg_cpp_isSubscribedTo(RWAvailabilityMsg_C *subscriber, void* source);

void RWAvailabilityMsg_C_addAuthor(RWAvailabilityMsg_C *coowner, RWAvailabilityMsg_C *data);

void RWAvailabilityMsg_C_init(RWAvailabilityMsg_C *owner);

int RWAvailabilityMsg_C_isLinked(RWAvailabilityMsg_C *data);

int RWAvailabilityMsg_C_isWritten(RWAvailabilityMsg_C *data);

uint64_t RWAvailabilityMsg_C_timeWritten(RWAvailabilityMsg_C *data);

int64_t RWAvailabilityMsg_C_moduleID(RWAvailabilityMsg_C *data);

void RWAvailabilityMsg_C_write(RWAvailabilityMsgPayload *data, RWAvailabilityMsg_C *destination, int64_t moduleID, uint64_t callTime);

RWAvailabilityMsgPayload RWAvailabilityMsg_C_read(RWAvailabilityMsg_C *source);

RWAvailabilityMsgPayload RWAvailabilityMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif