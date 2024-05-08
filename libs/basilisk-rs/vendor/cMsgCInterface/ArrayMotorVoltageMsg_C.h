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

#ifndef ArrayMotorVoltageMsg_C_H
#define ArrayMotorVoltageMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ArrayMotorVoltageMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ArrayMotorVoltageMsgPayload payload;		        //!< message copy, zero'd on construction
    ArrayMotorVoltageMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ArrayMotorVoltageMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ArrayMotorVoltageMsg_cpp_subscribe(ArrayMotorVoltageMsg_C *subscriber, void* source);

void ArrayMotorVoltageMsg_C_subscribe(ArrayMotorVoltageMsg_C *subscriber, ArrayMotorVoltageMsg_C *source);

int8_t ArrayMotorVoltageMsg_C_isSubscribedTo(ArrayMotorVoltageMsg_C *subscriber, ArrayMotorVoltageMsg_C *source);
int8_t ArrayMotorVoltageMsg_cpp_isSubscribedTo(ArrayMotorVoltageMsg_C *subscriber, void* source);

void ArrayMotorVoltageMsg_C_addAuthor(ArrayMotorVoltageMsg_C *coowner, ArrayMotorVoltageMsg_C *data);

void ArrayMotorVoltageMsg_C_init(ArrayMotorVoltageMsg_C *owner);

int ArrayMotorVoltageMsg_C_isLinked(ArrayMotorVoltageMsg_C *data);

int ArrayMotorVoltageMsg_C_isWritten(ArrayMotorVoltageMsg_C *data);

uint64_t ArrayMotorVoltageMsg_C_timeWritten(ArrayMotorVoltageMsg_C *data);

int64_t ArrayMotorVoltageMsg_C_moduleID(ArrayMotorVoltageMsg_C *data);

void ArrayMotorVoltageMsg_C_write(ArrayMotorVoltageMsgPayload *data, ArrayMotorVoltageMsg_C *destination, int64_t moduleID, uint64_t callTime);

ArrayMotorVoltageMsgPayload ArrayMotorVoltageMsg_C_read(ArrayMotorVoltageMsg_C *source);

ArrayMotorVoltageMsgPayload ArrayMotorVoltageMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif