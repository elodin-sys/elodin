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

#ifndef CSSArraySensorMsg_C_H
#define CSSArraySensorMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CSSArraySensorMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CSSArraySensorMsgPayload payload;		        //!< message copy, zero'd on construction
    CSSArraySensorMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CSSArraySensorMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CSSArraySensorMsg_cpp_subscribe(CSSArraySensorMsg_C *subscriber, void* source);

void CSSArraySensorMsg_C_subscribe(CSSArraySensorMsg_C *subscriber, CSSArraySensorMsg_C *source);

int8_t CSSArraySensorMsg_C_isSubscribedTo(CSSArraySensorMsg_C *subscriber, CSSArraySensorMsg_C *source);
int8_t CSSArraySensorMsg_cpp_isSubscribedTo(CSSArraySensorMsg_C *subscriber, void* source);

void CSSArraySensorMsg_C_addAuthor(CSSArraySensorMsg_C *coowner, CSSArraySensorMsg_C *data);

void CSSArraySensorMsg_C_init(CSSArraySensorMsg_C *owner);

int CSSArraySensorMsg_C_isLinked(CSSArraySensorMsg_C *data);

int CSSArraySensorMsg_C_isWritten(CSSArraySensorMsg_C *data);

uint64_t CSSArraySensorMsg_C_timeWritten(CSSArraySensorMsg_C *data);

int64_t CSSArraySensorMsg_C_moduleID(CSSArraySensorMsg_C *data);

void CSSArraySensorMsg_C_write(CSSArraySensorMsgPayload *data, CSSArraySensorMsg_C *destination, int64_t moduleID, uint64_t callTime);

CSSArraySensorMsgPayload CSSArraySensorMsg_C_read(CSSArraySensorMsg_C *source);

CSSArraySensorMsgPayload CSSArraySensorMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif