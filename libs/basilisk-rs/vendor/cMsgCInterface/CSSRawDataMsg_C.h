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

#ifndef CSSRawDataMsg_C_H
#define CSSRawDataMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CSSRawDataMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CSSRawDataMsgPayload payload;		        //!< message copy, zero'd on construction
    CSSRawDataMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CSSRawDataMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CSSRawDataMsg_cpp_subscribe(CSSRawDataMsg_C *subscriber, void* source);

void CSSRawDataMsg_C_subscribe(CSSRawDataMsg_C *subscriber, CSSRawDataMsg_C *source);

int8_t CSSRawDataMsg_C_isSubscribedTo(CSSRawDataMsg_C *subscriber, CSSRawDataMsg_C *source);
int8_t CSSRawDataMsg_cpp_isSubscribedTo(CSSRawDataMsg_C *subscriber, void* source);

void CSSRawDataMsg_C_addAuthor(CSSRawDataMsg_C *coowner, CSSRawDataMsg_C *data);

void CSSRawDataMsg_C_init(CSSRawDataMsg_C *owner);

int CSSRawDataMsg_C_isLinked(CSSRawDataMsg_C *data);

int CSSRawDataMsg_C_isWritten(CSSRawDataMsg_C *data);

uint64_t CSSRawDataMsg_C_timeWritten(CSSRawDataMsg_C *data);

int64_t CSSRawDataMsg_C_moduleID(CSSRawDataMsg_C *data);

void CSSRawDataMsg_C_write(CSSRawDataMsgPayload *data, CSSRawDataMsg_C *destination, int64_t moduleID, uint64_t callTime);

CSSRawDataMsgPayload CSSRawDataMsg_C_read(CSSRawDataMsg_C *source);

CSSRawDataMsgPayload CSSRawDataMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif