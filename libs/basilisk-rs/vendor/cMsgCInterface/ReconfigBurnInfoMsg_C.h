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

#ifndef ReconfigBurnInfoMsg_C_H
#define ReconfigBurnInfoMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ReconfigBurnInfoMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ReconfigBurnInfoMsgPayload payload;		        //!< message copy, zero'd on construction
    ReconfigBurnInfoMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ReconfigBurnInfoMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ReconfigBurnInfoMsg_cpp_subscribe(ReconfigBurnInfoMsg_C *subscriber, void* source);

void ReconfigBurnInfoMsg_C_subscribe(ReconfigBurnInfoMsg_C *subscriber, ReconfigBurnInfoMsg_C *source);

int8_t ReconfigBurnInfoMsg_C_isSubscribedTo(ReconfigBurnInfoMsg_C *subscriber, ReconfigBurnInfoMsg_C *source);
int8_t ReconfigBurnInfoMsg_cpp_isSubscribedTo(ReconfigBurnInfoMsg_C *subscriber, void* source);

void ReconfigBurnInfoMsg_C_addAuthor(ReconfigBurnInfoMsg_C *coowner, ReconfigBurnInfoMsg_C *data);

void ReconfigBurnInfoMsg_C_init(ReconfigBurnInfoMsg_C *owner);

int ReconfigBurnInfoMsg_C_isLinked(ReconfigBurnInfoMsg_C *data);

int ReconfigBurnInfoMsg_C_isWritten(ReconfigBurnInfoMsg_C *data);

uint64_t ReconfigBurnInfoMsg_C_timeWritten(ReconfigBurnInfoMsg_C *data);

int64_t ReconfigBurnInfoMsg_C_moduleID(ReconfigBurnInfoMsg_C *data);

void ReconfigBurnInfoMsg_C_write(ReconfigBurnInfoMsgPayload *data, ReconfigBurnInfoMsg_C *destination, int64_t moduleID, uint64_t callTime);

ReconfigBurnInfoMsgPayload ReconfigBurnInfoMsg_C_read(ReconfigBurnInfoMsg_C *source);

ReconfigBurnInfoMsgPayload ReconfigBurnInfoMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif