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

#ifndef SunlineFilterMsg_C_H
#define SunlineFilterMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SunlineFilterMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SunlineFilterMsgPayload payload;		        //!< message copy, zero'd on construction
    SunlineFilterMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SunlineFilterMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SunlineFilterMsg_cpp_subscribe(SunlineFilterMsg_C *subscriber, void* source);

void SunlineFilterMsg_C_subscribe(SunlineFilterMsg_C *subscriber, SunlineFilterMsg_C *source);

int8_t SunlineFilterMsg_C_isSubscribedTo(SunlineFilterMsg_C *subscriber, SunlineFilterMsg_C *source);
int8_t SunlineFilterMsg_cpp_isSubscribedTo(SunlineFilterMsg_C *subscriber, void* source);

void SunlineFilterMsg_C_addAuthor(SunlineFilterMsg_C *coowner, SunlineFilterMsg_C *data);

void SunlineFilterMsg_C_init(SunlineFilterMsg_C *owner);

int SunlineFilterMsg_C_isLinked(SunlineFilterMsg_C *data);

int SunlineFilterMsg_C_isWritten(SunlineFilterMsg_C *data);

uint64_t SunlineFilterMsg_C_timeWritten(SunlineFilterMsg_C *data);

int64_t SunlineFilterMsg_C_moduleID(SunlineFilterMsg_C *data);

void SunlineFilterMsg_C_write(SunlineFilterMsgPayload *data, SunlineFilterMsg_C *destination, int64_t moduleID, uint64_t callTime);

SunlineFilterMsgPayload SunlineFilterMsg_C_read(SunlineFilterMsg_C *source);

SunlineFilterMsgPayload SunlineFilterMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif