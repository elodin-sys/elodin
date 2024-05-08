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

#ifndef HingedRigidBodyMsg_C_H
#define HingedRigidBodyMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/HingedRigidBodyMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    HingedRigidBodyMsgPayload payload;		        //!< message copy, zero'd on construction
    HingedRigidBodyMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} HingedRigidBodyMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void HingedRigidBodyMsg_cpp_subscribe(HingedRigidBodyMsg_C *subscriber, void* source);

void HingedRigidBodyMsg_C_subscribe(HingedRigidBodyMsg_C *subscriber, HingedRigidBodyMsg_C *source);

int8_t HingedRigidBodyMsg_C_isSubscribedTo(HingedRigidBodyMsg_C *subscriber, HingedRigidBodyMsg_C *source);
int8_t HingedRigidBodyMsg_cpp_isSubscribedTo(HingedRigidBodyMsg_C *subscriber, void* source);

void HingedRigidBodyMsg_C_addAuthor(HingedRigidBodyMsg_C *coowner, HingedRigidBodyMsg_C *data);

void HingedRigidBodyMsg_C_init(HingedRigidBodyMsg_C *owner);

int HingedRigidBodyMsg_C_isLinked(HingedRigidBodyMsg_C *data);

int HingedRigidBodyMsg_C_isWritten(HingedRigidBodyMsg_C *data);

uint64_t HingedRigidBodyMsg_C_timeWritten(HingedRigidBodyMsg_C *data);

int64_t HingedRigidBodyMsg_C_moduleID(HingedRigidBodyMsg_C *data);

void HingedRigidBodyMsg_C_write(HingedRigidBodyMsgPayload *data, HingedRigidBodyMsg_C *destination, int64_t moduleID, uint64_t callTime);

HingedRigidBodyMsgPayload HingedRigidBodyMsg_C_read(HingedRigidBodyMsg_C *source);

HingedRigidBodyMsgPayload HingedRigidBodyMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif