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

#ifndef NavAttMsg_C_H
#define NavAttMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/NavAttMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    NavAttMsgPayload payload;		        //!< message copy, zero'd on construction
    NavAttMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} NavAttMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void NavAttMsg_cpp_subscribe(NavAttMsg_C *subscriber, void* source);

void NavAttMsg_C_subscribe(NavAttMsg_C *subscriber, NavAttMsg_C *source);

int8_t NavAttMsg_C_isSubscribedTo(NavAttMsg_C *subscriber, NavAttMsg_C *source);
int8_t NavAttMsg_cpp_isSubscribedTo(NavAttMsg_C *subscriber, void* source);

void NavAttMsg_C_addAuthor(NavAttMsg_C *coowner, NavAttMsg_C *data);

void NavAttMsg_C_init(NavAttMsg_C *owner);

int NavAttMsg_C_isLinked(NavAttMsg_C *data);

int NavAttMsg_C_isWritten(NavAttMsg_C *data);

uint64_t NavAttMsg_C_timeWritten(NavAttMsg_C *data);

int64_t NavAttMsg_C_moduleID(NavAttMsg_C *data);

void NavAttMsg_C_write(NavAttMsgPayload *data, NavAttMsg_C *destination, int64_t moduleID, uint64_t callTime);

NavAttMsgPayload NavAttMsg_C_read(NavAttMsg_C *source);

NavAttMsgPayload NavAttMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif