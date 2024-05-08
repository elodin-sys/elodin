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

#ifndef IMUSensorMsg_C_H
#define IMUSensorMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/IMUSensorMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    IMUSensorMsgPayload payload;		        //!< message copy, zero'd on construction
    IMUSensorMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} IMUSensorMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void IMUSensorMsg_cpp_subscribe(IMUSensorMsg_C *subscriber, void* source);

void IMUSensorMsg_C_subscribe(IMUSensorMsg_C *subscriber, IMUSensorMsg_C *source);

int8_t IMUSensorMsg_C_isSubscribedTo(IMUSensorMsg_C *subscriber, IMUSensorMsg_C *source);
int8_t IMUSensorMsg_cpp_isSubscribedTo(IMUSensorMsg_C *subscriber, void* source);

void IMUSensorMsg_C_addAuthor(IMUSensorMsg_C *coowner, IMUSensorMsg_C *data);

void IMUSensorMsg_C_init(IMUSensorMsg_C *owner);

int IMUSensorMsg_C_isLinked(IMUSensorMsg_C *data);

int IMUSensorMsg_C_isWritten(IMUSensorMsg_C *data);

uint64_t IMUSensorMsg_C_timeWritten(IMUSensorMsg_C *data);

int64_t IMUSensorMsg_C_moduleID(IMUSensorMsg_C *data);

void IMUSensorMsg_C_write(IMUSensorMsgPayload *data, IMUSensorMsg_C *destination, int64_t moduleID, uint64_t callTime);

IMUSensorMsgPayload IMUSensorMsg_C_read(IMUSensorMsg_C *source);

IMUSensorMsgPayload IMUSensorMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif