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

#ifndef OpNavMsg_C_H
#define OpNavMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/OpNavMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    OpNavMsgPayload payload;		        //!< message copy, zero'd on construction
    OpNavMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} OpNavMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void OpNavMsg_cpp_subscribe(OpNavMsg_C *subscriber, void* source);

void OpNavMsg_C_subscribe(OpNavMsg_C *subscriber, OpNavMsg_C *source);

int8_t OpNavMsg_C_isSubscribedTo(OpNavMsg_C *subscriber, OpNavMsg_C *source);
int8_t OpNavMsg_cpp_isSubscribedTo(OpNavMsg_C *subscriber, void* source);

void OpNavMsg_C_addAuthor(OpNavMsg_C *coowner, OpNavMsg_C *data);

void OpNavMsg_C_init(OpNavMsg_C *owner);

int OpNavMsg_C_isLinked(OpNavMsg_C *data);

int OpNavMsg_C_isWritten(OpNavMsg_C *data);

uint64_t OpNavMsg_C_timeWritten(OpNavMsg_C *data);

int64_t OpNavMsg_C_moduleID(OpNavMsg_C *data);

void OpNavMsg_C_write(OpNavMsgPayload *data, OpNavMsg_C *destination, int64_t moduleID, uint64_t callTime);

OpNavMsgPayload OpNavMsg_C_read(OpNavMsg_C *source);

OpNavMsgPayload OpNavMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif