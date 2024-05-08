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

#ifndef CameraConfigMsg_C_H
#define CameraConfigMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CameraConfigMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CameraConfigMsgPayload payload;		        //!< message copy, zero'd on construction
    CameraConfigMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CameraConfigMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CameraConfigMsg_cpp_subscribe(CameraConfigMsg_C *subscriber, void* source);

void CameraConfigMsg_C_subscribe(CameraConfigMsg_C *subscriber, CameraConfigMsg_C *source);

int8_t CameraConfigMsg_C_isSubscribedTo(CameraConfigMsg_C *subscriber, CameraConfigMsg_C *source);
int8_t CameraConfigMsg_cpp_isSubscribedTo(CameraConfigMsg_C *subscriber, void* source);

void CameraConfigMsg_C_addAuthor(CameraConfigMsg_C *coowner, CameraConfigMsg_C *data);

void CameraConfigMsg_C_init(CameraConfigMsg_C *owner);

int CameraConfigMsg_C_isLinked(CameraConfigMsg_C *data);

int CameraConfigMsg_C_isWritten(CameraConfigMsg_C *data);

uint64_t CameraConfigMsg_C_timeWritten(CameraConfigMsg_C *data);

int64_t CameraConfigMsg_C_moduleID(CameraConfigMsg_C *data);

void CameraConfigMsg_C_write(CameraConfigMsgPayload *data, CameraConfigMsg_C *destination, int64_t moduleID, uint64_t callTime);

CameraConfigMsgPayload CameraConfigMsg_C_read(CameraConfigMsg_C *source);

CameraConfigMsgPayload CameraConfigMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif