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

#ifndef SCEnergyMomentumMsg_C_H
#define SCEnergyMomentumMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SCEnergyMomentumMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SCEnergyMomentumMsgPayload payload;		        //!< message copy, zero'd on construction
    SCEnergyMomentumMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SCEnergyMomentumMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SCEnergyMomentumMsg_cpp_subscribe(SCEnergyMomentumMsg_C *subscriber, void* source);

void SCEnergyMomentumMsg_C_subscribe(SCEnergyMomentumMsg_C *subscriber, SCEnergyMomentumMsg_C *source);

int8_t SCEnergyMomentumMsg_C_isSubscribedTo(SCEnergyMomentumMsg_C *subscriber, SCEnergyMomentumMsg_C *source);
int8_t SCEnergyMomentumMsg_cpp_isSubscribedTo(SCEnergyMomentumMsg_C *subscriber, void* source);

void SCEnergyMomentumMsg_C_addAuthor(SCEnergyMomentumMsg_C *coowner, SCEnergyMomentumMsg_C *data);

void SCEnergyMomentumMsg_C_init(SCEnergyMomentumMsg_C *owner);

int SCEnergyMomentumMsg_C_isLinked(SCEnergyMomentumMsg_C *data);

int SCEnergyMomentumMsg_C_isWritten(SCEnergyMomentumMsg_C *data);

uint64_t SCEnergyMomentumMsg_C_timeWritten(SCEnergyMomentumMsg_C *data);

int64_t SCEnergyMomentumMsg_C_moduleID(SCEnergyMomentumMsg_C *data);

void SCEnergyMomentumMsg_C_write(SCEnergyMomentumMsgPayload *data, SCEnergyMomentumMsg_C *destination, int64_t moduleID, uint64_t callTime);

SCEnergyMomentumMsgPayload SCEnergyMomentumMsg_C_read(SCEnergyMomentumMsg_C *source);

SCEnergyMomentumMsgPayload SCEnergyMomentumMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif