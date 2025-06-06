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

#ifndef PixelLineFilterMsg_C_H
#define PixelLineFilterMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PixelLineFilterMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PixelLineFilterMsgPayload payload;		        //!< message copy, zero'd on construction
    PixelLineFilterMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PixelLineFilterMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PixelLineFilterMsg_cpp_subscribe(PixelLineFilterMsg_C *subscriber, void* source);

void PixelLineFilterMsg_C_subscribe(PixelLineFilterMsg_C *subscriber, PixelLineFilterMsg_C *source);

int8_t PixelLineFilterMsg_C_isSubscribedTo(PixelLineFilterMsg_C *subscriber, PixelLineFilterMsg_C *source);
int8_t PixelLineFilterMsg_cpp_isSubscribedTo(PixelLineFilterMsg_C *subscriber, void* source);

void PixelLineFilterMsg_C_addAuthor(PixelLineFilterMsg_C *coowner, PixelLineFilterMsg_C *data);

void PixelLineFilterMsg_C_init(PixelLineFilterMsg_C *owner);

int PixelLineFilterMsg_C_isLinked(PixelLineFilterMsg_C *data);

int PixelLineFilterMsg_C_isWritten(PixelLineFilterMsg_C *data);

uint64_t PixelLineFilterMsg_C_timeWritten(PixelLineFilterMsg_C *data);

int64_t PixelLineFilterMsg_C_moduleID(PixelLineFilterMsg_C *data);

void PixelLineFilterMsg_C_write(PixelLineFilterMsgPayload *data, PixelLineFilterMsg_C *destination, int64_t moduleID, uint64_t callTime);

PixelLineFilterMsgPayload PixelLineFilterMsg_C_read(PixelLineFilterMsg_C *source);

PixelLineFilterMsgPayload PixelLineFilterMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif