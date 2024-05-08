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

#ifndef ArrayMotorForceMsg_C_H
#define ArrayMotorForceMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ArrayMotorForceMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ArrayMotorForceMsgPayload payload;		        //!< message copy, zero'd on construction
    ArrayMotorForceMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ArrayMotorForceMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ArrayMotorForceMsg_cpp_subscribe(ArrayMotorForceMsg_C *subscriber, void* source);

void ArrayMotorForceMsg_C_subscribe(ArrayMotorForceMsg_C *subscriber, ArrayMotorForceMsg_C *source);

int8_t ArrayMotorForceMsg_C_isSubscribedTo(ArrayMotorForceMsg_C *subscriber, ArrayMotorForceMsg_C *source);
int8_t ArrayMotorForceMsg_cpp_isSubscribedTo(ArrayMotorForceMsg_C *subscriber, void* source);

void ArrayMotorForceMsg_C_addAuthor(ArrayMotorForceMsg_C *coowner, ArrayMotorForceMsg_C *data);

void ArrayMotorForceMsg_C_init(ArrayMotorForceMsg_C *owner);

int ArrayMotorForceMsg_C_isLinked(ArrayMotorForceMsg_C *data);

int ArrayMotorForceMsg_C_isWritten(ArrayMotorForceMsg_C *data);

uint64_t ArrayMotorForceMsg_C_timeWritten(ArrayMotorForceMsg_C *data);

int64_t ArrayMotorForceMsg_C_moduleID(ArrayMotorForceMsg_C *data);

void ArrayMotorForceMsg_C_write(ArrayMotorForceMsgPayload *data, ArrayMotorForceMsg_C *destination, int64_t moduleID, uint64_t callTime);

ArrayMotorForceMsgPayload ArrayMotorForceMsg_C_read(ArrayMotorForceMsg_C *source);

ArrayMotorForceMsgPayload ArrayMotorForceMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif