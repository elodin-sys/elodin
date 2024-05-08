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

#ifndef NavTransMsg_C_H
#define NavTransMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/NavTransMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    NavTransMsgPayload payload;		        //!< message copy, zero'd on construction
    NavTransMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} NavTransMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void NavTransMsg_cpp_subscribe(NavTransMsg_C *subscriber, void* source);

void NavTransMsg_C_subscribe(NavTransMsg_C *subscriber, NavTransMsg_C *source);

int8_t NavTransMsg_C_isSubscribedTo(NavTransMsg_C *subscriber, NavTransMsg_C *source);
int8_t NavTransMsg_cpp_isSubscribedTo(NavTransMsg_C *subscriber, void* source);

void NavTransMsg_C_addAuthor(NavTransMsg_C *coowner, NavTransMsg_C *data);

void NavTransMsg_C_init(NavTransMsg_C *owner);

int NavTransMsg_C_isLinked(NavTransMsg_C *data);

int NavTransMsg_C_isWritten(NavTransMsg_C *data);

uint64_t NavTransMsg_C_timeWritten(NavTransMsg_C *data);

int64_t NavTransMsg_C_moduleID(NavTransMsg_C *data);

void NavTransMsg_C_write(NavTransMsgPayload *data, NavTransMsg_C *destination, int64_t moduleID, uint64_t callTime);

NavTransMsgPayload NavTransMsg_C_read(NavTransMsg_C *source);

NavTransMsgPayload NavTransMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif