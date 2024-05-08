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

#ifndef LandmarkMsg_C_H
#define LandmarkMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/LandmarkMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    LandmarkMsgPayload payload;		        //!< message copy, zero'd on construction
    LandmarkMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} LandmarkMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void LandmarkMsg_cpp_subscribe(LandmarkMsg_C *subscriber, void* source);

void LandmarkMsg_C_subscribe(LandmarkMsg_C *subscriber, LandmarkMsg_C *source);

int8_t LandmarkMsg_C_isSubscribedTo(LandmarkMsg_C *subscriber, LandmarkMsg_C *source);
int8_t LandmarkMsg_cpp_isSubscribedTo(LandmarkMsg_C *subscriber, void* source);

void LandmarkMsg_C_addAuthor(LandmarkMsg_C *coowner, LandmarkMsg_C *data);

void LandmarkMsg_C_init(LandmarkMsg_C *owner);

int LandmarkMsg_C_isLinked(LandmarkMsg_C *data);

int LandmarkMsg_C_isWritten(LandmarkMsg_C *data);

uint64_t LandmarkMsg_C_timeWritten(LandmarkMsg_C *data);

int64_t LandmarkMsg_C_moduleID(LandmarkMsg_C *data);

void LandmarkMsg_C_write(LandmarkMsgPayload *data, LandmarkMsg_C *destination, int64_t moduleID, uint64_t callTime);

LandmarkMsgPayload LandmarkMsg_C_read(LandmarkMsg_C *source);

LandmarkMsgPayload LandmarkMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif