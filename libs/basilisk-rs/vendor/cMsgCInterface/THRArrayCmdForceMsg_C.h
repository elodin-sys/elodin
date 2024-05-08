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

#ifndef THRArrayCmdForceMsg_C_H
#define THRArrayCmdForceMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/THRArrayCmdForceMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    THRArrayCmdForceMsgPayload payload;		        //!< message copy, zero'd on construction
    THRArrayCmdForceMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} THRArrayCmdForceMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void THRArrayCmdForceMsg_cpp_subscribe(THRArrayCmdForceMsg_C *subscriber, void* source);

void THRArrayCmdForceMsg_C_subscribe(THRArrayCmdForceMsg_C *subscriber, THRArrayCmdForceMsg_C *source);

int8_t THRArrayCmdForceMsg_C_isSubscribedTo(THRArrayCmdForceMsg_C *subscriber, THRArrayCmdForceMsg_C *source);
int8_t THRArrayCmdForceMsg_cpp_isSubscribedTo(THRArrayCmdForceMsg_C *subscriber, void* source);

void THRArrayCmdForceMsg_C_addAuthor(THRArrayCmdForceMsg_C *coowner, THRArrayCmdForceMsg_C *data);

void THRArrayCmdForceMsg_C_init(THRArrayCmdForceMsg_C *owner);

int THRArrayCmdForceMsg_C_isLinked(THRArrayCmdForceMsg_C *data);

int THRArrayCmdForceMsg_C_isWritten(THRArrayCmdForceMsg_C *data);

uint64_t THRArrayCmdForceMsg_C_timeWritten(THRArrayCmdForceMsg_C *data);

int64_t THRArrayCmdForceMsg_C_moduleID(THRArrayCmdForceMsg_C *data);

void THRArrayCmdForceMsg_C_write(THRArrayCmdForceMsgPayload *data, THRArrayCmdForceMsg_C *destination, int64_t moduleID, uint64_t callTime);

THRArrayCmdForceMsgPayload THRArrayCmdForceMsg_C_read(THRArrayCmdForceMsg_C *source);

THRArrayCmdForceMsgPayload THRArrayCmdForceMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif