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

#ifndef SpiceTimeMsg_C_H
#define SpiceTimeMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SpiceTimeMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SpiceTimeMsgPayload payload;		        //!< message copy, zero'd on construction
    SpiceTimeMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SpiceTimeMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SpiceTimeMsg_cpp_subscribe(SpiceTimeMsg_C *subscriber, void* source);

void SpiceTimeMsg_C_subscribe(SpiceTimeMsg_C *subscriber, SpiceTimeMsg_C *source);

int8_t SpiceTimeMsg_C_isSubscribedTo(SpiceTimeMsg_C *subscriber, SpiceTimeMsg_C *source);
int8_t SpiceTimeMsg_cpp_isSubscribedTo(SpiceTimeMsg_C *subscriber, void* source);

void SpiceTimeMsg_C_addAuthor(SpiceTimeMsg_C *coowner, SpiceTimeMsg_C *data);

void SpiceTimeMsg_C_init(SpiceTimeMsg_C *owner);

int SpiceTimeMsg_C_isLinked(SpiceTimeMsg_C *data);

int SpiceTimeMsg_C_isWritten(SpiceTimeMsg_C *data);

uint64_t SpiceTimeMsg_C_timeWritten(SpiceTimeMsg_C *data);

int64_t SpiceTimeMsg_C_moduleID(SpiceTimeMsg_C *data);

void SpiceTimeMsg_C_write(SpiceTimeMsgPayload *data, SpiceTimeMsg_C *destination, int64_t moduleID, uint64_t callTime);

SpiceTimeMsgPayload SpiceTimeMsg_C_read(SpiceTimeMsg_C *source);

SpiceTimeMsgPayload SpiceTimeMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif