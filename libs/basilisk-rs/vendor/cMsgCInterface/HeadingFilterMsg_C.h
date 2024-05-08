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

#ifndef HeadingFilterMsg_C_H
#define HeadingFilterMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/HeadingFilterMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    HeadingFilterMsgPayload payload;		        //!< message copy, zero'd on construction
    HeadingFilterMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} HeadingFilterMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void HeadingFilterMsg_cpp_subscribe(HeadingFilterMsg_C *subscriber, void* source);

void HeadingFilterMsg_C_subscribe(HeadingFilterMsg_C *subscriber, HeadingFilterMsg_C *source);

int8_t HeadingFilterMsg_C_isSubscribedTo(HeadingFilterMsg_C *subscriber, HeadingFilterMsg_C *source);
int8_t HeadingFilterMsg_cpp_isSubscribedTo(HeadingFilterMsg_C *subscriber, void* source);

void HeadingFilterMsg_C_addAuthor(HeadingFilterMsg_C *coowner, HeadingFilterMsg_C *data);

void HeadingFilterMsg_C_init(HeadingFilterMsg_C *owner);

int HeadingFilterMsg_C_isLinked(HeadingFilterMsg_C *data);

int HeadingFilterMsg_C_isWritten(HeadingFilterMsg_C *data);

uint64_t HeadingFilterMsg_C_timeWritten(HeadingFilterMsg_C *data);

int64_t HeadingFilterMsg_C_moduleID(HeadingFilterMsg_C *data);

void HeadingFilterMsg_C_write(HeadingFilterMsgPayload *data, HeadingFilterMsg_C *destination, int64_t moduleID, uint64_t callTime);

HeadingFilterMsgPayload HeadingFilterMsg_C_read(HeadingFilterMsg_C *source);

HeadingFilterMsgPayload HeadingFilterMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif