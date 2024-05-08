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

#ifndef PowerStorageStatusMsg_C_H
#define PowerStorageStatusMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PowerStorageStatusMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PowerStorageStatusMsgPayload payload;		        //!< message copy, zero'd on construction
    PowerStorageStatusMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PowerStorageStatusMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PowerStorageStatusMsg_cpp_subscribe(PowerStorageStatusMsg_C *subscriber, void* source);

void PowerStorageStatusMsg_C_subscribe(PowerStorageStatusMsg_C *subscriber, PowerStorageStatusMsg_C *source);

int8_t PowerStorageStatusMsg_C_isSubscribedTo(PowerStorageStatusMsg_C *subscriber, PowerStorageStatusMsg_C *source);
int8_t PowerStorageStatusMsg_cpp_isSubscribedTo(PowerStorageStatusMsg_C *subscriber, void* source);

void PowerStorageStatusMsg_C_addAuthor(PowerStorageStatusMsg_C *coowner, PowerStorageStatusMsg_C *data);

void PowerStorageStatusMsg_C_init(PowerStorageStatusMsg_C *owner);

int PowerStorageStatusMsg_C_isLinked(PowerStorageStatusMsg_C *data);

int PowerStorageStatusMsg_C_isWritten(PowerStorageStatusMsg_C *data);

uint64_t PowerStorageStatusMsg_C_timeWritten(PowerStorageStatusMsg_C *data);

int64_t PowerStorageStatusMsg_C_moduleID(PowerStorageStatusMsg_C *data);

void PowerStorageStatusMsg_C_write(PowerStorageStatusMsgPayload *data, PowerStorageStatusMsg_C *destination, int64_t moduleID, uint64_t callTime);

PowerStorageStatusMsgPayload PowerStorageStatusMsg_C_read(PowerStorageStatusMsg_C *source);

PowerStorageStatusMsgPayload PowerStorageStatusMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif