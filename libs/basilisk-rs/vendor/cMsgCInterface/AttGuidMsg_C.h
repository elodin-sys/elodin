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

#ifndef AttGuidMsg_C_H
#define AttGuidMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AttGuidMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AttGuidMsgPayload payload;		        //!< message copy, zero'd on construction
    AttGuidMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AttGuidMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AttGuidMsg_cpp_subscribe(AttGuidMsg_C *subscriber, void* source);

void AttGuidMsg_C_subscribe(AttGuidMsg_C *subscriber, AttGuidMsg_C *source);

int8_t AttGuidMsg_C_isSubscribedTo(AttGuidMsg_C *subscriber, AttGuidMsg_C *source);
int8_t AttGuidMsg_cpp_isSubscribedTo(AttGuidMsg_C *subscriber, void* source);

void AttGuidMsg_C_addAuthor(AttGuidMsg_C *coowner, AttGuidMsg_C *data);

void AttGuidMsg_C_init(AttGuidMsg_C *owner);

int AttGuidMsg_C_isLinked(AttGuidMsg_C *data);

int AttGuidMsg_C_isWritten(AttGuidMsg_C *data);

uint64_t AttGuidMsg_C_timeWritten(AttGuidMsg_C *data);

int64_t AttGuidMsg_C_moduleID(AttGuidMsg_C *data);

void AttGuidMsg_C_write(AttGuidMsgPayload *data, AttGuidMsg_C *destination, int64_t moduleID, uint64_t callTime);

AttGuidMsgPayload AttGuidMsg_C_read(AttGuidMsg_C *source);

AttGuidMsgPayload AttGuidMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif