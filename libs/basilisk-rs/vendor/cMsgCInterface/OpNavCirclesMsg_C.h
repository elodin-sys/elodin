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

#ifndef OpNavCirclesMsg_C_H
#define OpNavCirclesMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    OpNavCirclesMsgPayload payload;		        //!< message copy, zero'd on construction
    OpNavCirclesMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} OpNavCirclesMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void OpNavCirclesMsg_cpp_subscribe(OpNavCirclesMsg_C *subscriber, void* source);

void OpNavCirclesMsg_C_subscribe(OpNavCirclesMsg_C *subscriber, OpNavCirclesMsg_C *source);

int8_t OpNavCirclesMsg_C_isSubscribedTo(OpNavCirclesMsg_C *subscriber, OpNavCirclesMsg_C *source);
int8_t OpNavCirclesMsg_cpp_isSubscribedTo(OpNavCirclesMsg_C *subscriber, void* source);

void OpNavCirclesMsg_C_addAuthor(OpNavCirclesMsg_C *coowner, OpNavCirclesMsg_C *data);

void OpNavCirclesMsg_C_init(OpNavCirclesMsg_C *owner);

int OpNavCirclesMsg_C_isLinked(OpNavCirclesMsg_C *data);

int OpNavCirclesMsg_C_isWritten(OpNavCirclesMsg_C *data);

uint64_t OpNavCirclesMsg_C_timeWritten(OpNavCirclesMsg_C *data);

int64_t OpNavCirclesMsg_C_moduleID(OpNavCirclesMsg_C *data);

void OpNavCirclesMsg_C_write(OpNavCirclesMsgPayload *data, OpNavCirclesMsg_C *destination, int64_t moduleID, uint64_t callTime);

OpNavCirclesMsgPayload OpNavCirclesMsg_C_read(OpNavCirclesMsg_C *source);

OpNavCirclesMsgPayload OpNavCirclesMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif