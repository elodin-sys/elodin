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

#ifndef PrescribedRotationMsg_C_H
#define PrescribedRotationMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PrescribedRotationMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PrescribedRotationMsgPayload payload;		        //!< message copy, zero'd on construction
    PrescribedRotationMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PrescribedRotationMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PrescribedRotationMsg_cpp_subscribe(PrescribedRotationMsg_C *subscriber, void* source);

void PrescribedRotationMsg_C_subscribe(PrescribedRotationMsg_C *subscriber, PrescribedRotationMsg_C *source);

int8_t PrescribedRotationMsg_C_isSubscribedTo(PrescribedRotationMsg_C *subscriber, PrescribedRotationMsg_C *source);
int8_t PrescribedRotationMsg_cpp_isSubscribedTo(PrescribedRotationMsg_C *subscriber, void* source);

void PrescribedRotationMsg_C_addAuthor(PrescribedRotationMsg_C *coowner, PrescribedRotationMsg_C *data);

void PrescribedRotationMsg_C_init(PrescribedRotationMsg_C *owner);

int PrescribedRotationMsg_C_isLinked(PrescribedRotationMsg_C *data);

int PrescribedRotationMsg_C_isWritten(PrescribedRotationMsg_C *data);

uint64_t PrescribedRotationMsg_C_timeWritten(PrescribedRotationMsg_C *data);

int64_t PrescribedRotationMsg_C_moduleID(PrescribedRotationMsg_C *data);

void PrescribedRotationMsg_C_write(PrescribedRotationMsgPayload *data, PrescribedRotationMsg_C *destination, int64_t moduleID, uint64_t callTime);

PrescribedRotationMsgPayload PrescribedRotationMsg_C_read(PrescribedRotationMsg_C *source);

PrescribedRotationMsgPayload PrescribedRotationMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif