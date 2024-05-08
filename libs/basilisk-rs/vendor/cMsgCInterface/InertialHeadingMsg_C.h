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

#ifndef InertialHeadingMsg_C_H
#define InertialHeadingMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/InertialHeadingMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    InertialHeadingMsgPayload payload;		        //!< message copy, zero'd on construction
    InertialHeadingMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} InertialHeadingMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void InertialHeadingMsg_cpp_subscribe(InertialHeadingMsg_C *subscriber, void* source);

void InertialHeadingMsg_C_subscribe(InertialHeadingMsg_C *subscriber, InertialHeadingMsg_C *source);

int8_t InertialHeadingMsg_C_isSubscribedTo(InertialHeadingMsg_C *subscriber, InertialHeadingMsg_C *source);
int8_t InertialHeadingMsg_cpp_isSubscribedTo(InertialHeadingMsg_C *subscriber, void* source);

void InertialHeadingMsg_C_addAuthor(InertialHeadingMsg_C *coowner, InertialHeadingMsg_C *data);

void InertialHeadingMsg_C_init(InertialHeadingMsg_C *owner);

int InertialHeadingMsg_C_isLinked(InertialHeadingMsg_C *data);

int InertialHeadingMsg_C_isWritten(InertialHeadingMsg_C *data);

uint64_t InertialHeadingMsg_C_timeWritten(InertialHeadingMsg_C *data);

int64_t InertialHeadingMsg_C_moduleID(InertialHeadingMsg_C *data);

void InertialHeadingMsg_C_write(InertialHeadingMsgPayload *data, InertialHeadingMsg_C *destination, int64_t moduleID, uint64_t callTime);

InertialHeadingMsgPayload InertialHeadingMsg_C_read(InertialHeadingMsg_C *source);

InertialHeadingMsgPayload InertialHeadingMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif