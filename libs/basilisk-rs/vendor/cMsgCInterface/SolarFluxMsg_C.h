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

#ifndef SolarFluxMsg_C_H
#define SolarFluxMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SolarFluxMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SolarFluxMsgPayload payload;		        //!< message copy, zero'd on construction
    SolarFluxMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SolarFluxMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SolarFluxMsg_cpp_subscribe(SolarFluxMsg_C *subscriber, void* source);

void SolarFluxMsg_C_subscribe(SolarFluxMsg_C *subscriber, SolarFluxMsg_C *source);

int8_t SolarFluxMsg_C_isSubscribedTo(SolarFluxMsg_C *subscriber, SolarFluxMsg_C *source);
int8_t SolarFluxMsg_cpp_isSubscribedTo(SolarFluxMsg_C *subscriber, void* source);

void SolarFluxMsg_C_addAuthor(SolarFluxMsg_C *coowner, SolarFluxMsg_C *data);

void SolarFluxMsg_C_init(SolarFluxMsg_C *owner);

int SolarFluxMsg_C_isLinked(SolarFluxMsg_C *data);

int SolarFluxMsg_C_isWritten(SolarFluxMsg_C *data);

uint64_t SolarFluxMsg_C_timeWritten(SolarFluxMsg_C *data);

int64_t SolarFluxMsg_C_moduleID(SolarFluxMsg_C *data);

void SolarFluxMsg_C_write(SolarFluxMsgPayload *data, SolarFluxMsg_C *destination, int64_t moduleID, uint64_t callTime);

SolarFluxMsgPayload SolarFluxMsg_C_read(SolarFluxMsg_C *source);

SolarFluxMsgPayload SolarFluxMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif