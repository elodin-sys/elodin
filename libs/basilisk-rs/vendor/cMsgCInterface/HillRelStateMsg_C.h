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

#ifndef HillRelStateMsg_C_H
#define HillRelStateMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/HillRelStateMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    HillRelStateMsgPayload payload;		        //!< message copy, zero'd on construction
    HillRelStateMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} HillRelStateMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void HillRelStateMsg_cpp_subscribe(HillRelStateMsg_C *subscriber, void* source);

void HillRelStateMsg_C_subscribe(HillRelStateMsg_C *subscriber, HillRelStateMsg_C *source);

int8_t HillRelStateMsg_C_isSubscribedTo(HillRelStateMsg_C *subscriber, HillRelStateMsg_C *source);
int8_t HillRelStateMsg_cpp_isSubscribedTo(HillRelStateMsg_C *subscriber, void* source);

void HillRelStateMsg_C_addAuthor(HillRelStateMsg_C *coowner, HillRelStateMsg_C *data);

void HillRelStateMsg_C_init(HillRelStateMsg_C *owner);

int HillRelStateMsg_C_isLinked(HillRelStateMsg_C *data);

int HillRelStateMsg_C_isWritten(HillRelStateMsg_C *data);

uint64_t HillRelStateMsg_C_timeWritten(HillRelStateMsg_C *data);

int64_t HillRelStateMsg_C_moduleID(HillRelStateMsg_C *data);

void HillRelStateMsg_C_write(HillRelStateMsgPayload *data, HillRelStateMsg_C *destination, int64_t moduleID, uint64_t callTime);

HillRelStateMsgPayload HillRelStateMsg_C_read(HillRelStateMsg_C *source);

HillRelStateMsgPayload HillRelStateMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif