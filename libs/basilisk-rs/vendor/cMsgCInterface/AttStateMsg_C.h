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

#ifndef AttStateMsg_C_H
#define AttStateMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AttStateMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AttStateMsgPayload payload;		        //!< message copy, zero'd on construction
    AttStateMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AttStateMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AttStateMsg_cpp_subscribe(AttStateMsg_C *subscriber, void* source);

void AttStateMsg_C_subscribe(AttStateMsg_C *subscriber, AttStateMsg_C *source);

int8_t AttStateMsg_C_isSubscribedTo(AttStateMsg_C *subscriber, AttStateMsg_C *source);
int8_t AttStateMsg_cpp_isSubscribedTo(AttStateMsg_C *subscriber, void* source);

void AttStateMsg_C_addAuthor(AttStateMsg_C *coowner, AttStateMsg_C *data);

void AttStateMsg_C_init(AttStateMsg_C *owner);

int AttStateMsg_C_isLinked(AttStateMsg_C *data);

int AttStateMsg_C_isWritten(AttStateMsg_C *data);

uint64_t AttStateMsg_C_timeWritten(AttStateMsg_C *data);

int64_t AttStateMsg_C_moduleID(AttStateMsg_C *data);

void AttStateMsg_C_write(AttStateMsgPayload *data, AttStateMsg_C *destination, int64_t moduleID, uint64_t callTime);

AttStateMsgPayload AttStateMsg_C_read(AttStateMsg_C *source);

AttStateMsgPayload AttStateMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif