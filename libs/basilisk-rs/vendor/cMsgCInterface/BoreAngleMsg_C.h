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

#ifndef BoreAngleMsg_C_H
#define BoreAngleMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/BoreAngleMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    BoreAngleMsgPayload payload;		        //!< message copy, zero'd on construction
    BoreAngleMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} BoreAngleMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void BoreAngleMsg_cpp_subscribe(BoreAngleMsg_C *subscriber, void* source);

void BoreAngleMsg_C_subscribe(BoreAngleMsg_C *subscriber, BoreAngleMsg_C *source);

int8_t BoreAngleMsg_C_isSubscribedTo(BoreAngleMsg_C *subscriber, BoreAngleMsg_C *source);
int8_t BoreAngleMsg_cpp_isSubscribedTo(BoreAngleMsg_C *subscriber, void* source);

void BoreAngleMsg_C_addAuthor(BoreAngleMsg_C *coowner, BoreAngleMsg_C *data);

void BoreAngleMsg_C_init(BoreAngleMsg_C *owner);

int BoreAngleMsg_C_isLinked(BoreAngleMsg_C *data);

int BoreAngleMsg_C_isWritten(BoreAngleMsg_C *data);

uint64_t BoreAngleMsg_C_timeWritten(BoreAngleMsg_C *data);

int64_t BoreAngleMsg_C_moduleID(BoreAngleMsg_C *data);

void BoreAngleMsg_C_write(BoreAngleMsgPayload *data, BoreAngleMsg_C *destination, int64_t moduleID, uint64_t callTime);

BoreAngleMsgPayload BoreAngleMsg_C_read(BoreAngleMsg_C *source);

BoreAngleMsgPayload BoreAngleMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif