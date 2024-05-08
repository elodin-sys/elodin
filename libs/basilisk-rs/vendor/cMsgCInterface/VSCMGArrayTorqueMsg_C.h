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

#ifndef VSCMGArrayTorqueMsg_C_H
#define VSCMGArrayTorqueMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/VSCMGArrayTorqueMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    VSCMGArrayTorqueMsgPayload payload;		        //!< message copy, zero'd on construction
    VSCMGArrayTorqueMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} VSCMGArrayTorqueMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void VSCMGArrayTorqueMsg_cpp_subscribe(VSCMGArrayTorqueMsg_C *subscriber, void* source);

void VSCMGArrayTorqueMsg_C_subscribe(VSCMGArrayTorqueMsg_C *subscriber, VSCMGArrayTorqueMsg_C *source);

int8_t VSCMGArrayTorqueMsg_C_isSubscribedTo(VSCMGArrayTorqueMsg_C *subscriber, VSCMGArrayTorqueMsg_C *source);
int8_t VSCMGArrayTorqueMsg_cpp_isSubscribedTo(VSCMGArrayTorqueMsg_C *subscriber, void* source);

void VSCMGArrayTorqueMsg_C_addAuthor(VSCMGArrayTorqueMsg_C *coowner, VSCMGArrayTorqueMsg_C *data);

void VSCMGArrayTorqueMsg_C_init(VSCMGArrayTorqueMsg_C *owner);

int VSCMGArrayTorqueMsg_C_isLinked(VSCMGArrayTorqueMsg_C *data);

int VSCMGArrayTorqueMsg_C_isWritten(VSCMGArrayTorqueMsg_C *data);

uint64_t VSCMGArrayTorqueMsg_C_timeWritten(VSCMGArrayTorqueMsg_C *data);

int64_t VSCMGArrayTorqueMsg_C_moduleID(VSCMGArrayTorqueMsg_C *data);

void VSCMGArrayTorqueMsg_C_write(VSCMGArrayTorqueMsgPayload *data, VSCMGArrayTorqueMsg_C *destination, int64_t moduleID, uint64_t callTime);

VSCMGArrayTorqueMsgPayload VSCMGArrayTorqueMsg_C_read(VSCMGArrayTorqueMsg_C *source);

VSCMGArrayTorqueMsgPayload VSCMGArrayTorqueMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif