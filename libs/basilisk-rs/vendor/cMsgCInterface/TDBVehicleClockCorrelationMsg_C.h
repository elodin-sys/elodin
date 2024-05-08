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

#ifndef TDBVehicleClockCorrelationMsg_C_H
#define TDBVehicleClockCorrelationMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/TDBVehicleClockCorrelationMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    TDBVehicleClockCorrelationMsgPayload payload;		        //!< message copy, zero'd on construction
    TDBVehicleClockCorrelationMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} TDBVehicleClockCorrelationMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void TDBVehicleClockCorrelationMsg_cpp_subscribe(TDBVehicleClockCorrelationMsg_C *subscriber, void* source);

void TDBVehicleClockCorrelationMsg_C_subscribe(TDBVehicleClockCorrelationMsg_C *subscriber, TDBVehicleClockCorrelationMsg_C *source);

int8_t TDBVehicleClockCorrelationMsg_C_isSubscribedTo(TDBVehicleClockCorrelationMsg_C *subscriber, TDBVehicleClockCorrelationMsg_C *source);
int8_t TDBVehicleClockCorrelationMsg_cpp_isSubscribedTo(TDBVehicleClockCorrelationMsg_C *subscriber, void* source);

void TDBVehicleClockCorrelationMsg_C_addAuthor(TDBVehicleClockCorrelationMsg_C *coowner, TDBVehicleClockCorrelationMsg_C *data);

void TDBVehicleClockCorrelationMsg_C_init(TDBVehicleClockCorrelationMsg_C *owner);

int TDBVehicleClockCorrelationMsg_C_isLinked(TDBVehicleClockCorrelationMsg_C *data);

int TDBVehicleClockCorrelationMsg_C_isWritten(TDBVehicleClockCorrelationMsg_C *data);

uint64_t TDBVehicleClockCorrelationMsg_C_timeWritten(TDBVehicleClockCorrelationMsg_C *data);

int64_t TDBVehicleClockCorrelationMsg_C_moduleID(TDBVehicleClockCorrelationMsg_C *data);

void TDBVehicleClockCorrelationMsg_C_write(TDBVehicleClockCorrelationMsgPayload *data, TDBVehicleClockCorrelationMsg_C *destination, int64_t moduleID, uint64_t callTime);

TDBVehicleClockCorrelationMsgPayload TDBVehicleClockCorrelationMsg_C_read(TDBVehicleClockCorrelationMsg_C *source);

TDBVehicleClockCorrelationMsgPayload TDBVehicleClockCorrelationMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif