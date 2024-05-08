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

#ifndef DeviceStatusMsg_C_H
#define DeviceStatusMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/DeviceStatusMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    DeviceStatusMsgPayload payload;		        //!< message copy, zero'd on construction
    DeviceStatusMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} DeviceStatusMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void DeviceStatusMsg_cpp_subscribe(DeviceStatusMsg_C *subscriber, void* source);

void DeviceStatusMsg_C_subscribe(DeviceStatusMsg_C *subscriber, DeviceStatusMsg_C *source);

int8_t DeviceStatusMsg_C_isSubscribedTo(DeviceStatusMsg_C *subscriber, DeviceStatusMsg_C *source);
int8_t DeviceStatusMsg_cpp_isSubscribedTo(DeviceStatusMsg_C *subscriber, void* source);

void DeviceStatusMsg_C_addAuthor(DeviceStatusMsg_C *coowner, DeviceStatusMsg_C *data);

void DeviceStatusMsg_C_init(DeviceStatusMsg_C *owner);

int DeviceStatusMsg_C_isLinked(DeviceStatusMsg_C *data);

int DeviceStatusMsg_C_isWritten(DeviceStatusMsg_C *data);

uint64_t DeviceStatusMsg_C_timeWritten(DeviceStatusMsg_C *data);

int64_t DeviceStatusMsg_C_moduleID(DeviceStatusMsg_C *data);

void DeviceStatusMsg_C_write(DeviceStatusMsgPayload *data, DeviceStatusMsg_C *destination, int64_t moduleID, uint64_t callTime);

DeviceStatusMsgPayload DeviceStatusMsg_C_read(DeviceStatusMsg_C *source);

DeviceStatusMsgPayload DeviceStatusMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif