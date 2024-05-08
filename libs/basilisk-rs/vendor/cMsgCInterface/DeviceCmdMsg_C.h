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

#ifndef DeviceCmdMsg_C_H
#define DeviceCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/DeviceCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    DeviceCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    DeviceCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} DeviceCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void DeviceCmdMsg_cpp_subscribe(DeviceCmdMsg_C *subscriber, void* source);

void DeviceCmdMsg_C_subscribe(DeviceCmdMsg_C *subscriber, DeviceCmdMsg_C *source);

int8_t DeviceCmdMsg_C_isSubscribedTo(DeviceCmdMsg_C *subscriber, DeviceCmdMsg_C *source);
int8_t DeviceCmdMsg_cpp_isSubscribedTo(DeviceCmdMsg_C *subscriber, void* source);

void DeviceCmdMsg_C_addAuthor(DeviceCmdMsg_C *coowner, DeviceCmdMsg_C *data);

void DeviceCmdMsg_C_init(DeviceCmdMsg_C *owner);

int DeviceCmdMsg_C_isLinked(DeviceCmdMsg_C *data);

int DeviceCmdMsg_C_isWritten(DeviceCmdMsg_C *data);

uint64_t DeviceCmdMsg_C_timeWritten(DeviceCmdMsg_C *data);

int64_t DeviceCmdMsg_C_moduleID(DeviceCmdMsg_C *data);

void DeviceCmdMsg_C_write(DeviceCmdMsgPayload *data, DeviceCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

DeviceCmdMsgPayload DeviceCmdMsg_C_read(DeviceCmdMsg_C *source);

DeviceCmdMsgPayload DeviceCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif