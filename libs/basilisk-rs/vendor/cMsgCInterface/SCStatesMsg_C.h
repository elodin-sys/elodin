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

#ifndef SCStatesMsg_C_H
#define SCStatesMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SCStatesMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SCStatesMsgPayload payload;		        //!< message copy, zero'd on construction
    SCStatesMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SCStatesMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SCStatesMsg_cpp_subscribe(SCStatesMsg_C *subscriber, void* source);

void SCStatesMsg_C_subscribe(SCStatesMsg_C *subscriber, SCStatesMsg_C *source);

int8_t SCStatesMsg_C_isSubscribedTo(SCStatesMsg_C *subscriber, SCStatesMsg_C *source);
int8_t SCStatesMsg_cpp_isSubscribedTo(SCStatesMsg_C *subscriber, void* source);

void SCStatesMsg_C_addAuthor(SCStatesMsg_C *coowner, SCStatesMsg_C *data);

void SCStatesMsg_C_init(SCStatesMsg_C *owner);

int SCStatesMsg_C_isLinked(SCStatesMsg_C *data);

int SCStatesMsg_C_isWritten(SCStatesMsg_C *data);

uint64_t SCStatesMsg_C_timeWritten(SCStatesMsg_C *data);

int64_t SCStatesMsg_C_moduleID(SCStatesMsg_C *data);

void SCStatesMsg_C_write(SCStatesMsgPayload *data, SCStatesMsg_C *destination, int64_t moduleID, uint64_t callTime);

SCStatesMsgPayload SCStatesMsg_C_read(SCStatesMsg_C *source);

SCStatesMsgPayload SCStatesMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif