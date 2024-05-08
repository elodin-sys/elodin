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

#ifndef CmdForceInertialMsg_C_H
#define CmdForceInertialMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/CmdForceInertialMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    CmdForceInertialMsgPayload payload;		        //!< message copy, zero'd on construction
    CmdForceInertialMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} CmdForceInertialMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void CmdForceInertialMsg_cpp_subscribe(CmdForceInertialMsg_C *subscriber, void* source);

void CmdForceInertialMsg_C_subscribe(CmdForceInertialMsg_C *subscriber, CmdForceInertialMsg_C *source);

int8_t CmdForceInertialMsg_C_isSubscribedTo(CmdForceInertialMsg_C *subscriber, CmdForceInertialMsg_C *source);
int8_t CmdForceInertialMsg_cpp_isSubscribedTo(CmdForceInertialMsg_C *subscriber, void* source);

void CmdForceInertialMsg_C_addAuthor(CmdForceInertialMsg_C *coowner, CmdForceInertialMsg_C *data);

void CmdForceInertialMsg_C_init(CmdForceInertialMsg_C *owner);

int CmdForceInertialMsg_C_isLinked(CmdForceInertialMsg_C *data);

int CmdForceInertialMsg_C_isWritten(CmdForceInertialMsg_C *data);

uint64_t CmdForceInertialMsg_C_timeWritten(CmdForceInertialMsg_C *data);

int64_t CmdForceInertialMsg_C_moduleID(CmdForceInertialMsg_C *data);

void CmdForceInertialMsg_C_write(CmdForceInertialMsgPayload *data, CmdForceInertialMsg_C *destination, int64_t moduleID, uint64_t callTime);

CmdForceInertialMsgPayload CmdForceInertialMsg_C_read(CmdForceInertialMsg_C *source);

CmdForceInertialMsgPayload CmdForceInertialMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif