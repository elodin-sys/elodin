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

#ifndef SpicePlanetStateMsg_C_H
#define SpicePlanetStateMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SpicePlanetStateMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SpicePlanetStateMsgPayload payload;		        //!< message copy, zero'd on construction
    SpicePlanetStateMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SpicePlanetStateMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SpicePlanetStateMsg_cpp_subscribe(SpicePlanetStateMsg_C *subscriber, void* source);

void SpicePlanetStateMsg_C_subscribe(SpicePlanetStateMsg_C *subscriber, SpicePlanetStateMsg_C *source);

int8_t SpicePlanetStateMsg_C_isSubscribedTo(SpicePlanetStateMsg_C *subscriber, SpicePlanetStateMsg_C *source);
int8_t SpicePlanetStateMsg_cpp_isSubscribedTo(SpicePlanetStateMsg_C *subscriber, void* source);

void SpicePlanetStateMsg_C_addAuthor(SpicePlanetStateMsg_C *coowner, SpicePlanetStateMsg_C *data);

void SpicePlanetStateMsg_C_init(SpicePlanetStateMsg_C *owner);

int SpicePlanetStateMsg_C_isLinked(SpicePlanetStateMsg_C *data);

int SpicePlanetStateMsg_C_isWritten(SpicePlanetStateMsg_C *data);

uint64_t SpicePlanetStateMsg_C_timeWritten(SpicePlanetStateMsg_C *data);

int64_t SpicePlanetStateMsg_C_moduleID(SpicePlanetStateMsg_C *data);

void SpicePlanetStateMsg_C_write(SpicePlanetStateMsgPayload *data, SpicePlanetStateMsg_C *destination, int64_t moduleID, uint64_t callTime);

SpicePlanetStateMsgPayload SpicePlanetStateMsg_C_read(SpicePlanetStateMsg_C *source);

SpicePlanetStateMsgPayload SpicePlanetStateMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif