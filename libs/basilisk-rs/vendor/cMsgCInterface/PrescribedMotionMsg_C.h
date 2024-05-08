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

#ifndef PrescribedMotionMsg_C_H
#define PrescribedMotionMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PrescribedMotionMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PrescribedMotionMsgPayload payload;		        //!< message copy, zero'd on construction
    PrescribedMotionMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PrescribedMotionMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PrescribedMotionMsg_cpp_subscribe(PrescribedMotionMsg_C *subscriber, void* source);

void PrescribedMotionMsg_C_subscribe(PrescribedMotionMsg_C *subscriber, PrescribedMotionMsg_C *source);

int8_t PrescribedMotionMsg_C_isSubscribedTo(PrescribedMotionMsg_C *subscriber, PrescribedMotionMsg_C *source);
int8_t PrescribedMotionMsg_cpp_isSubscribedTo(PrescribedMotionMsg_C *subscriber, void* source);

void PrescribedMotionMsg_C_addAuthor(PrescribedMotionMsg_C *coowner, PrescribedMotionMsg_C *data);

void PrescribedMotionMsg_C_init(PrescribedMotionMsg_C *owner);

int PrescribedMotionMsg_C_isLinked(PrescribedMotionMsg_C *data);

int PrescribedMotionMsg_C_isWritten(PrescribedMotionMsg_C *data);

uint64_t PrescribedMotionMsg_C_timeWritten(PrescribedMotionMsg_C *data);

int64_t PrescribedMotionMsg_C_moduleID(PrescribedMotionMsg_C *data);

void PrescribedMotionMsg_C_write(PrescribedMotionMsgPayload *data, PrescribedMotionMsg_C *destination, int64_t moduleID, uint64_t callTime);

PrescribedMotionMsgPayload PrescribedMotionMsg_C_read(PrescribedMotionMsg_C *source);

PrescribedMotionMsgPayload PrescribedMotionMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif