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

#ifndef CmdForceBodyMsg_C_H
#define CmdForceBodyMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CmdForceBodyMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CmdForceBodyMsgPayload payload;		        //!< message copy, zero'd on construction
    CmdForceBodyMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CmdForceBodyMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CmdForceBodyMsg_cpp_subscribe(CmdForceBodyMsg_C *subscriber, void* source);

void CmdForceBodyMsg_C_subscribe(CmdForceBodyMsg_C *subscriber, CmdForceBodyMsg_C *source);

int8_t CmdForceBodyMsg_C_isSubscribedTo(CmdForceBodyMsg_C *subscriber, CmdForceBodyMsg_C *source);
int8_t CmdForceBodyMsg_cpp_isSubscribedTo(CmdForceBodyMsg_C *subscriber, void* source);

void CmdForceBodyMsg_C_addAuthor(CmdForceBodyMsg_C *coowner, CmdForceBodyMsg_C *data);

void CmdForceBodyMsg_C_init(CmdForceBodyMsg_C *owner);

int CmdForceBodyMsg_C_isLinked(CmdForceBodyMsg_C *data);

int CmdForceBodyMsg_C_isWritten(CmdForceBodyMsg_C *data);

uint64_t CmdForceBodyMsg_C_timeWritten(CmdForceBodyMsg_C *data);

int64_t CmdForceBodyMsg_C_moduleID(CmdForceBodyMsg_C *data);

void CmdForceBodyMsg_C_write(CmdForceBodyMsgPayload *data, CmdForceBodyMsg_C *destination, int64_t moduleID, uint64_t callTime);

CmdForceBodyMsgPayload CmdForceBodyMsg_C_read(CmdForceBodyMsg_C *source);

CmdForceBodyMsgPayload CmdForceBodyMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif