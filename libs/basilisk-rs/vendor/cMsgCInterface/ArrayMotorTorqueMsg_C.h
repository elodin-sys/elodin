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

#ifndef ArrayMotorTorqueMsg_C_H
#define ArrayMotorTorqueMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ArrayMotorTorqueMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ArrayMotorTorqueMsgPayload payload;		        //!< message copy, zero'd on construction
    ArrayMotorTorqueMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ArrayMotorTorqueMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ArrayMotorTorqueMsg_cpp_subscribe(ArrayMotorTorqueMsg_C *subscriber, void* source);

void ArrayMotorTorqueMsg_C_subscribe(ArrayMotorTorqueMsg_C *subscriber, ArrayMotorTorqueMsg_C *source);

int8_t ArrayMotorTorqueMsg_C_isSubscribedTo(ArrayMotorTorqueMsg_C *subscriber, ArrayMotorTorqueMsg_C *source);
int8_t ArrayMotorTorqueMsg_cpp_isSubscribedTo(ArrayMotorTorqueMsg_C *subscriber, void* source);

void ArrayMotorTorqueMsg_C_addAuthor(ArrayMotorTorqueMsg_C *coowner, ArrayMotorTorqueMsg_C *data);

void ArrayMotorTorqueMsg_C_init(ArrayMotorTorqueMsg_C *owner);

int ArrayMotorTorqueMsg_C_isLinked(ArrayMotorTorqueMsg_C *data);

int ArrayMotorTorqueMsg_C_isWritten(ArrayMotorTorqueMsg_C *data);

uint64_t ArrayMotorTorqueMsg_C_timeWritten(ArrayMotorTorqueMsg_C *data);

int64_t ArrayMotorTorqueMsg_C_moduleID(ArrayMotorTorqueMsg_C *data);

void ArrayMotorTorqueMsg_C_write(ArrayMotorTorqueMsgPayload *data, ArrayMotorTorqueMsg_C *destination, int64_t moduleID, uint64_t callTime);

ArrayMotorTorqueMsgPayload ArrayMotorTorqueMsg_C_read(ArrayMotorTorqueMsg_C *source);

ArrayMotorTorqueMsgPayload ArrayMotorTorqueMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif