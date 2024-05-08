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

#ifndef PlasmaFluxMsg_C_H
#define PlasmaFluxMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PlasmaFluxMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PlasmaFluxMsgPayload payload;		        //!< message copy, zero'd on construction
    PlasmaFluxMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PlasmaFluxMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PlasmaFluxMsg_cpp_subscribe(PlasmaFluxMsg_C *subscriber, void* source);

void PlasmaFluxMsg_C_subscribe(PlasmaFluxMsg_C *subscriber, PlasmaFluxMsg_C *source);

int8_t PlasmaFluxMsg_C_isSubscribedTo(PlasmaFluxMsg_C *subscriber, PlasmaFluxMsg_C *source);
int8_t PlasmaFluxMsg_cpp_isSubscribedTo(PlasmaFluxMsg_C *subscriber, void* source);

void PlasmaFluxMsg_C_addAuthor(PlasmaFluxMsg_C *coowner, PlasmaFluxMsg_C *data);

void PlasmaFluxMsg_C_init(PlasmaFluxMsg_C *owner);

int PlasmaFluxMsg_C_isLinked(PlasmaFluxMsg_C *data);

int PlasmaFluxMsg_C_isWritten(PlasmaFluxMsg_C *data);

uint64_t PlasmaFluxMsg_C_timeWritten(PlasmaFluxMsg_C *data);

int64_t PlasmaFluxMsg_C_moduleID(PlasmaFluxMsg_C *data);

void PlasmaFluxMsg_C_write(PlasmaFluxMsgPayload *data, PlasmaFluxMsg_C *destination, int64_t moduleID, uint64_t callTime);

PlasmaFluxMsgPayload PlasmaFluxMsg_C_read(PlasmaFluxMsg_C *source);

PlasmaFluxMsgPayload PlasmaFluxMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif