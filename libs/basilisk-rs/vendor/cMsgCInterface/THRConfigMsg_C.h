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

#ifndef THRConfigMsg_C_H
#define THRConfigMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/THRConfigMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    THRConfigMsgPayload payload;		        //!< message copy, zero'd on construction
    THRConfigMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} THRConfigMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void THRConfigMsg_cpp_subscribe(THRConfigMsg_C *subscriber, void* source);

void THRConfigMsg_C_subscribe(THRConfigMsg_C *subscriber, THRConfigMsg_C *source);

int8_t THRConfigMsg_C_isSubscribedTo(THRConfigMsg_C *subscriber, THRConfigMsg_C *source);
int8_t THRConfigMsg_cpp_isSubscribedTo(THRConfigMsg_C *subscriber, void* source);

void THRConfigMsg_C_addAuthor(THRConfigMsg_C *coowner, THRConfigMsg_C *data);

void THRConfigMsg_C_init(THRConfigMsg_C *owner);

int THRConfigMsg_C_isLinked(THRConfigMsg_C *data);

int THRConfigMsg_C_isWritten(THRConfigMsg_C *data);

uint64_t THRConfigMsg_C_timeWritten(THRConfigMsg_C *data);

int64_t THRConfigMsg_C_moduleID(THRConfigMsg_C *data);

void THRConfigMsg_C_write(THRConfigMsgPayload *data, THRConfigMsg_C *destination, int64_t moduleID, uint64_t callTime);

THRConfigMsgPayload THRConfigMsg_C_read(THRConfigMsg_C *source);

THRConfigMsgPayload THRConfigMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif