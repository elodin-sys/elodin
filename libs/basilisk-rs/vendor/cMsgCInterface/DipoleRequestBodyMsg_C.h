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

#ifndef DipoleRequestBodyMsg_C_H
#define DipoleRequestBodyMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/DipoleRequestBodyMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    DipoleRequestBodyMsgPayload payload;		        //!< message copy, zero'd on construction
    DipoleRequestBodyMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} DipoleRequestBodyMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void DipoleRequestBodyMsg_cpp_subscribe(DipoleRequestBodyMsg_C *subscriber, void* source);

void DipoleRequestBodyMsg_C_subscribe(DipoleRequestBodyMsg_C *subscriber, DipoleRequestBodyMsg_C *source);

int8_t DipoleRequestBodyMsg_C_isSubscribedTo(DipoleRequestBodyMsg_C *subscriber, DipoleRequestBodyMsg_C *source);
int8_t DipoleRequestBodyMsg_cpp_isSubscribedTo(DipoleRequestBodyMsg_C *subscriber, void* source);

void DipoleRequestBodyMsg_C_addAuthor(DipoleRequestBodyMsg_C *coowner, DipoleRequestBodyMsg_C *data);

void DipoleRequestBodyMsg_C_init(DipoleRequestBodyMsg_C *owner);

int DipoleRequestBodyMsg_C_isLinked(DipoleRequestBodyMsg_C *data);

int DipoleRequestBodyMsg_C_isWritten(DipoleRequestBodyMsg_C *data);

uint64_t DipoleRequestBodyMsg_C_timeWritten(DipoleRequestBodyMsg_C *data);

int64_t DipoleRequestBodyMsg_C_moduleID(DipoleRequestBodyMsg_C *data);

void DipoleRequestBodyMsg_C_write(DipoleRequestBodyMsgPayload *data, DipoleRequestBodyMsg_C *destination, int64_t moduleID, uint64_t callTime);

DipoleRequestBodyMsgPayload DipoleRequestBodyMsg_C_read(DipoleRequestBodyMsg_C *source);

DipoleRequestBodyMsgPayload DipoleRequestBodyMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif