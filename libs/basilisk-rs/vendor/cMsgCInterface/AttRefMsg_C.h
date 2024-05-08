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

#ifndef AttRefMsg_C_H
#define AttRefMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AttRefMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AttRefMsgPayload payload;		        //!< message copy, zero'd on construction
    AttRefMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AttRefMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AttRefMsg_cpp_subscribe(AttRefMsg_C *subscriber, void* source);

void AttRefMsg_C_subscribe(AttRefMsg_C *subscriber, AttRefMsg_C *source);

int8_t AttRefMsg_C_isSubscribedTo(AttRefMsg_C *subscriber, AttRefMsg_C *source);
int8_t AttRefMsg_cpp_isSubscribedTo(AttRefMsg_C *subscriber, void* source);

void AttRefMsg_C_addAuthor(AttRefMsg_C *coowner, AttRefMsg_C *data);

void AttRefMsg_C_init(AttRefMsg_C *owner);

int AttRefMsg_C_isLinked(AttRefMsg_C *data);

int AttRefMsg_C_isWritten(AttRefMsg_C *data);

uint64_t AttRefMsg_C_timeWritten(AttRefMsg_C *data);

int64_t AttRefMsg_C_moduleID(AttRefMsg_C *data);

void AttRefMsg_C_write(AttRefMsgPayload *data, AttRefMsg_C *destination, int64_t moduleID, uint64_t callTime);

AttRefMsgPayload AttRefMsg_C_read(AttRefMsg_C *source);

AttRefMsgPayload AttRefMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif