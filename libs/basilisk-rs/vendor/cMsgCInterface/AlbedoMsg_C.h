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

#ifndef AlbedoMsg_C_H
#define AlbedoMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AlbedoMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AlbedoMsgPayload payload;		        //!< message copy, zero'd on construction
    AlbedoMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AlbedoMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AlbedoMsg_cpp_subscribe(AlbedoMsg_C *subscriber, void* source);

void AlbedoMsg_C_subscribe(AlbedoMsg_C *subscriber, AlbedoMsg_C *source);

int8_t AlbedoMsg_C_isSubscribedTo(AlbedoMsg_C *subscriber, AlbedoMsg_C *source);
int8_t AlbedoMsg_cpp_isSubscribedTo(AlbedoMsg_C *subscriber, void* source);

void AlbedoMsg_C_addAuthor(AlbedoMsg_C *coowner, AlbedoMsg_C *data);

void AlbedoMsg_C_init(AlbedoMsg_C *owner);

int AlbedoMsg_C_isLinked(AlbedoMsg_C *data);

int AlbedoMsg_C_isWritten(AlbedoMsg_C *data);

uint64_t AlbedoMsg_C_timeWritten(AlbedoMsg_C *data);

int64_t AlbedoMsg_C_moduleID(AlbedoMsg_C *data);

void AlbedoMsg_C_write(AlbedoMsgPayload *data, AlbedoMsg_C *destination, int64_t moduleID, uint64_t callTime);

AlbedoMsgPayload AlbedoMsg_C_read(AlbedoMsg_C *source);

AlbedoMsgPayload AlbedoMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif