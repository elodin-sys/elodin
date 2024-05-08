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

#ifndef ArrayEffectorLockMsg_C_H
#define ArrayEffectorLockMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ArrayEffectorLockMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ArrayEffectorLockMsgPayload payload;		        //!< message copy, zero'd on construction
    ArrayEffectorLockMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ArrayEffectorLockMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ArrayEffectorLockMsg_cpp_subscribe(ArrayEffectorLockMsg_C *subscriber, void* source);

void ArrayEffectorLockMsg_C_subscribe(ArrayEffectorLockMsg_C *subscriber, ArrayEffectorLockMsg_C *source);

int8_t ArrayEffectorLockMsg_C_isSubscribedTo(ArrayEffectorLockMsg_C *subscriber, ArrayEffectorLockMsg_C *source);
int8_t ArrayEffectorLockMsg_cpp_isSubscribedTo(ArrayEffectorLockMsg_C *subscriber, void* source);

void ArrayEffectorLockMsg_C_addAuthor(ArrayEffectorLockMsg_C *coowner, ArrayEffectorLockMsg_C *data);

void ArrayEffectorLockMsg_C_init(ArrayEffectorLockMsg_C *owner);

int ArrayEffectorLockMsg_C_isLinked(ArrayEffectorLockMsg_C *data);

int ArrayEffectorLockMsg_C_isWritten(ArrayEffectorLockMsg_C *data);

uint64_t ArrayEffectorLockMsg_C_timeWritten(ArrayEffectorLockMsg_C *data);

int64_t ArrayEffectorLockMsg_C_moduleID(ArrayEffectorLockMsg_C *data);

void ArrayEffectorLockMsg_C_write(ArrayEffectorLockMsgPayload *data, ArrayEffectorLockMsg_C *destination, int64_t moduleID, uint64_t callTime);

ArrayEffectorLockMsgPayload ArrayEffectorLockMsg_C_read(ArrayEffectorLockMsg_C *source);

ArrayEffectorLockMsgPayload ArrayEffectorLockMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif