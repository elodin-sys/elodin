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

#ifndef LinearTranslationRigidBodyMsg_C_H
#define LinearTranslationRigidBodyMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/LinearTranslationRigidBodyMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    LinearTranslationRigidBodyMsgPayload payload;		        //!< message copy, zero'd on construction
    LinearTranslationRigidBodyMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} LinearTranslationRigidBodyMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void LinearTranslationRigidBodyMsg_cpp_subscribe(LinearTranslationRigidBodyMsg_C *subscriber, void* source);

void LinearTranslationRigidBodyMsg_C_subscribe(LinearTranslationRigidBodyMsg_C *subscriber, LinearTranslationRigidBodyMsg_C *source);

int8_t LinearTranslationRigidBodyMsg_C_isSubscribedTo(LinearTranslationRigidBodyMsg_C *subscriber, LinearTranslationRigidBodyMsg_C *source);
int8_t LinearTranslationRigidBodyMsg_cpp_isSubscribedTo(LinearTranslationRigidBodyMsg_C *subscriber, void* source);

void LinearTranslationRigidBodyMsg_C_addAuthor(LinearTranslationRigidBodyMsg_C *coowner, LinearTranslationRigidBodyMsg_C *data);

void LinearTranslationRigidBodyMsg_C_init(LinearTranslationRigidBodyMsg_C *owner);

int LinearTranslationRigidBodyMsg_C_isLinked(LinearTranslationRigidBodyMsg_C *data);

int LinearTranslationRigidBodyMsg_C_isWritten(LinearTranslationRigidBodyMsg_C *data);

uint64_t LinearTranslationRigidBodyMsg_C_timeWritten(LinearTranslationRigidBodyMsg_C *data);

int64_t LinearTranslationRigidBodyMsg_C_moduleID(LinearTranslationRigidBodyMsg_C *data);

void LinearTranslationRigidBodyMsg_C_write(LinearTranslationRigidBodyMsgPayload *data, LinearTranslationRigidBodyMsg_C *destination, int64_t moduleID, uint64_t callTime);

LinearTranslationRigidBodyMsgPayload LinearTranslationRigidBodyMsg_C_read(LinearTranslationRigidBodyMsg_C *source);

LinearTranslationRigidBodyMsgPayload LinearTranslationRigidBodyMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif