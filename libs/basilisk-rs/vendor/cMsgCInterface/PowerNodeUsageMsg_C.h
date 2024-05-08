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

#ifndef PowerNodeUsageMsg_C_H
#define PowerNodeUsageMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PowerNodeUsageMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PowerNodeUsageMsgPayload payload;		        //!< message copy, zero'd on construction
    PowerNodeUsageMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PowerNodeUsageMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PowerNodeUsageMsg_cpp_subscribe(PowerNodeUsageMsg_C *subscriber, void* source);

void PowerNodeUsageMsg_C_subscribe(PowerNodeUsageMsg_C *subscriber, PowerNodeUsageMsg_C *source);

int8_t PowerNodeUsageMsg_C_isSubscribedTo(PowerNodeUsageMsg_C *subscriber, PowerNodeUsageMsg_C *source);
int8_t PowerNodeUsageMsg_cpp_isSubscribedTo(PowerNodeUsageMsg_C *subscriber, void* source);

void PowerNodeUsageMsg_C_addAuthor(PowerNodeUsageMsg_C *coowner, PowerNodeUsageMsg_C *data);

void PowerNodeUsageMsg_C_init(PowerNodeUsageMsg_C *owner);

int PowerNodeUsageMsg_C_isLinked(PowerNodeUsageMsg_C *data);

int PowerNodeUsageMsg_C_isWritten(PowerNodeUsageMsg_C *data);

uint64_t PowerNodeUsageMsg_C_timeWritten(PowerNodeUsageMsg_C *data);

int64_t PowerNodeUsageMsg_C_moduleID(PowerNodeUsageMsg_C *data);

void PowerNodeUsageMsg_C_write(PowerNodeUsageMsgPayload *data, PowerNodeUsageMsg_C *destination, int64_t moduleID, uint64_t callTime);

PowerNodeUsageMsgPayload PowerNodeUsageMsg_C_read(PowerNodeUsageMsg_C *source);

PowerNodeUsageMsgPayload PowerNodeUsageMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif