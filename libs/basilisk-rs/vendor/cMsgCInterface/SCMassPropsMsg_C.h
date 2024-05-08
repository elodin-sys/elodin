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

#ifndef SCMassPropsMsg_C_H
#define SCMassPropsMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SCMassPropsMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SCMassPropsMsgPayload payload;		        //!< message copy, zero'd on construction
    SCMassPropsMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SCMassPropsMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SCMassPropsMsg_cpp_subscribe(SCMassPropsMsg_C *subscriber, void* source);

void SCMassPropsMsg_C_subscribe(SCMassPropsMsg_C *subscriber, SCMassPropsMsg_C *source);

int8_t SCMassPropsMsg_C_isSubscribedTo(SCMassPropsMsg_C *subscriber, SCMassPropsMsg_C *source);
int8_t SCMassPropsMsg_cpp_isSubscribedTo(SCMassPropsMsg_C *subscriber, void* source);

void SCMassPropsMsg_C_addAuthor(SCMassPropsMsg_C *coowner, SCMassPropsMsg_C *data);

void SCMassPropsMsg_C_init(SCMassPropsMsg_C *owner);

int SCMassPropsMsg_C_isLinked(SCMassPropsMsg_C *data);

int SCMassPropsMsg_C_isWritten(SCMassPropsMsg_C *data);

uint64_t SCMassPropsMsg_C_timeWritten(SCMassPropsMsg_C *data);

int64_t SCMassPropsMsg_C_moduleID(SCMassPropsMsg_C *data);

void SCMassPropsMsg_C_write(SCMassPropsMsgPayload *data, SCMassPropsMsg_C *destination, int64_t moduleID, uint64_t callTime);

SCMassPropsMsgPayload SCMassPropsMsg_C_read(SCMassPropsMsg_C *source);

SCMassPropsMsgPayload SCMassPropsMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif