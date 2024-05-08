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

#ifndef CSSUnitConfigMsg_C_H
#define CSSUnitConfigMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CSSUnitConfigMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CSSUnitConfigMsgPayload payload;		        //!< message copy, zero'd on construction
    CSSUnitConfigMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CSSUnitConfigMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CSSUnitConfigMsg_cpp_subscribe(CSSUnitConfigMsg_C *subscriber, void* source);

void CSSUnitConfigMsg_C_subscribe(CSSUnitConfigMsg_C *subscriber, CSSUnitConfigMsg_C *source);

int8_t CSSUnitConfigMsg_C_isSubscribedTo(CSSUnitConfigMsg_C *subscriber, CSSUnitConfigMsg_C *source);
int8_t CSSUnitConfigMsg_cpp_isSubscribedTo(CSSUnitConfigMsg_C *subscriber, void* source);

void CSSUnitConfigMsg_C_addAuthor(CSSUnitConfigMsg_C *coowner, CSSUnitConfigMsg_C *data);

void CSSUnitConfigMsg_C_init(CSSUnitConfigMsg_C *owner);

int CSSUnitConfigMsg_C_isLinked(CSSUnitConfigMsg_C *data);

int CSSUnitConfigMsg_C_isWritten(CSSUnitConfigMsg_C *data);

uint64_t CSSUnitConfigMsg_C_timeWritten(CSSUnitConfigMsg_C *data);

int64_t CSSUnitConfigMsg_C_moduleID(CSSUnitConfigMsg_C *data);

void CSSUnitConfigMsg_C_write(CSSUnitConfigMsgPayload *data, CSSUnitConfigMsg_C *destination, int64_t moduleID, uint64_t callTime);

CSSUnitConfigMsgPayload CSSUnitConfigMsg_C_read(CSSUnitConfigMsg_C *source);

CSSUnitConfigMsgPayload CSSUnitConfigMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif