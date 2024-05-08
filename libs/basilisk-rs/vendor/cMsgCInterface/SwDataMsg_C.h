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

#ifndef SwDataMsg_C_H
#define SwDataMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SwDataMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SwDataMsgPayload payload;		        //!< message copy, zero'd on construction
    SwDataMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SwDataMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SwDataMsg_cpp_subscribe(SwDataMsg_C *subscriber, void* source);

void SwDataMsg_C_subscribe(SwDataMsg_C *subscriber, SwDataMsg_C *source);

int8_t SwDataMsg_C_isSubscribedTo(SwDataMsg_C *subscriber, SwDataMsg_C *source);
int8_t SwDataMsg_cpp_isSubscribedTo(SwDataMsg_C *subscriber, void* source);

void SwDataMsg_C_addAuthor(SwDataMsg_C *coowner, SwDataMsg_C *data);

void SwDataMsg_C_init(SwDataMsg_C *owner);

int SwDataMsg_C_isLinked(SwDataMsg_C *data);

int SwDataMsg_C_isWritten(SwDataMsg_C *data);

uint64_t SwDataMsg_C_timeWritten(SwDataMsg_C *data);

int64_t SwDataMsg_C_moduleID(SwDataMsg_C *data);

void SwDataMsg_C_write(SwDataMsgPayload *data, SwDataMsg_C *destination, int64_t moduleID, uint64_t callTime);

SwDataMsgPayload SwDataMsg_C_read(SwDataMsg_C *source);

SwDataMsgPayload SwDataMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif