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

#ifndef FuelTankMsg_C_H
#define FuelTankMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/FuelTankMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    FuelTankMsgPayload payload;		        //!< message copy, zero'd on construction
    FuelTankMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} FuelTankMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void FuelTankMsg_cpp_subscribe(FuelTankMsg_C *subscriber, void* source);

void FuelTankMsg_C_subscribe(FuelTankMsg_C *subscriber, FuelTankMsg_C *source);

int8_t FuelTankMsg_C_isSubscribedTo(FuelTankMsg_C *subscriber, FuelTankMsg_C *source);
int8_t FuelTankMsg_cpp_isSubscribedTo(FuelTankMsg_C *subscriber, void* source);

void FuelTankMsg_C_addAuthor(FuelTankMsg_C *coowner, FuelTankMsg_C *data);

void FuelTankMsg_C_init(FuelTankMsg_C *owner);

int FuelTankMsg_C_isLinked(FuelTankMsg_C *data);

int FuelTankMsg_C_isWritten(FuelTankMsg_C *data);

uint64_t FuelTankMsg_C_timeWritten(FuelTankMsg_C *data);

int64_t FuelTankMsg_C_moduleID(FuelTankMsg_C *data);

void FuelTankMsg_C_write(FuelTankMsgPayload *data, FuelTankMsg_C *destination, int64_t moduleID, uint64_t callTime);

FuelTankMsgPayload FuelTankMsg_C_read(FuelTankMsg_C *source);

FuelTankMsgPayload FuelTankMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif