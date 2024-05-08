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

#ifndef THRArrayOnTimeCmdMsg_C_H
#define THRArrayOnTimeCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/THRArrayOnTimeCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    THRArrayOnTimeCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    THRArrayOnTimeCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} THRArrayOnTimeCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void THRArrayOnTimeCmdMsg_cpp_subscribe(THRArrayOnTimeCmdMsg_C *subscriber, void* source);

void THRArrayOnTimeCmdMsg_C_subscribe(THRArrayOnTimeCmdMsg_C *subscriber, THRArrayOnTimeCmdMsg_C *source);

int8_t THRArrayOnTimeCmdMsg_C_isSubscribedTo(THRArrayOnTimeCmdMsg_C *subscriber, THRArrayOnTimeCmdMsg_C *source);
int8_t THRArrayOnTimeCmdMsg_cpp_isSubscribedTo(THRArrayOnTimeCmdMsg_C *subscriber, void* source);

void THRArrayOnTimeCmdMsg_C_addAuthor(THRArrayOnTimeCmdMsg_C *coowner, THRArrayOnTimeCmdMsg_C *data);

void THRArrayOnTimeCmdMsg_C_init(THRArrayOnTimeCmdMsg_C *owner);

int THRArrayOnTimeCmdMsg_C_isLinked(THRArrayOnTimeCmdMsg_C *data);

int THRArrayOnTimeCmdMsg_C_isWritten(THRArrayOnTimeCmdMsg_C *data);

uint64_t THRArrayOnTimeCmdMsg_C_timeWritten(THRArrayOnTimeCmdMsg_C *data);

int64_t THRArrayOnTimeCmdMsg_C_moduleID(THRArrayOnTimeCmdMsg_C *data);

void THRArrayOnTimeCmdMsg_C_write(THRArrayOnTimeCmdMsgPayload *data, THRArrayOnTimeCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

THRArrayOnTimeCmdMsgPayload THRArrayOnTimeCmdMsg_C_read(THRArrayOnTimeCmdMsg_C *source);

THRArrayOnTimeCmdMsgPayload THRArrayOnTimeCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif