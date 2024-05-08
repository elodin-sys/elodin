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

#ifndef VSCMGSpeedMsg_C_H
#define VSCMGSpeedMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/VSCMGSpeedMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    VSCMGSpeedMsgPayload payload;		        //!< message copy, zero'd on construction
    VSCMGSpeedMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} VSCMGSpeedMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void VSCMGSpeedMsg_cpp_subscribe(VSCMGSpeedMsg_C *subscriber, void* source);

void VSCMGSpeedMsg_C_subscribe(VSCMGSpeedMsg_C *subscriber, VSCMGSpeedMsg_C *source);

int8_t VSCMGSpeedMsg_C_isSubscribedTo(VSCMGSpeedMsg_C *subscriber, VSCMGSpeedMsg_C *source);
int8_t VSCMGSpeedMsg_cpp_isSubscribedTo(VSCMGSpeedMsg_C *subscriber, void* source);

void VSCMGSpeedMsg_C_addAuthor(VSCMGSpeedMsg_C *coowner, VSCMGSpeedMsg_C *data);

void VSCMGSpeedMsg_C_init(VSCMGSpeedMsg_C *owner);

int VSCMGSpeedMsg_C_isLinked(VSCMGSpeedMsg_C *data);

int VSCMGSpeedMsg_C_isWritten(VSCMGSpeedMsg_C *data);

uint64_t VSCMGSpeedMsg_C_timeWritten(VSCMGSpeedMsg_C *data);

int64_t VSCMGSpeedMsg_C_moduleID(VSCMGSpeedMsg_C *data);

void VSCMGSpeedMsg_C_write(VSCMGSpeedMsgPayload *data, VSCMGSpeedMsg_C *destination, int64_t moduleID, uint64_t callTime);

VSCMGSpeedMsgPayload VSCMGSpeedMsg_C_read(VSCMGSpeedMsg_C *source);

VSCMGSpeedMsgPayload VSCMGSpeedMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif