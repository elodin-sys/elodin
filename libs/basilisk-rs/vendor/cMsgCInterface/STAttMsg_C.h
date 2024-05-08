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

#ifndef STAttMsg_C_H
#define STAttMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/STAttMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    STAttMsgPayload payload;		        //!< message copy, zero'd on construction
    STAttMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} STAttMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void STAttMsg_cpp_subscribe(STAttMsg_C *subscriber, void* source);

void STAttMsg_C_subscribe(STAttMsg_C *subscriber, STAttMsg_C *source);

int8_t STAttMsg_C_isSubscribedTo(STAttMsg_C *subscriber, STAttMsg_C *source);
int8_t STAttMsg_cpp_isSubscribedTo(STAttMsg_C *subscriber, void* source);

void STAttMsg_C_addAuthor(STAttMsg_C *coowner, STAttMsg_C *data);

void STAttMsg_C_init(STAttMsg_C *owner);

int STAttMsg_C_isLinked(STAttMsg_C *data);

int STAttMsg_C_isWritten(STAttMsg_C *data);

uint64_t STAttMsg_C_timeWritten(STAttMsg_C *data);

int64_t STAttMsg_C_moduleID(STAttMsg_C *data);

void STAttMsg_C_write(STAttMsgPayload *data, STAttMsg_C *destination, int64_t moduleID, uint64_t callTime);

STAttMsgPayload STAttMsg_C_read(STAttMsg_C *source);

STAttMsgPayload STAttMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif