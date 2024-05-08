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

#ifndef SynchClockMsg_C_H
#define SynchClockMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/SynchClockMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    SynchClockMsgPayload payload;		        //!< message copy, zero'd on construction
    SynchClockMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} SynchClockMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void SynchClockMsg_cpp_subscribe(SynchClockMsg_C *subscriber, void* source);

void SynchClockMsg_C_subscribe(SynchClockMsg_C *subscriber, SynchClockMsg_C *source);

int8_t SynchClockMsg_C_isSubscribedTo(SynchClockMsg_C *subscriber, SynchClockMsg_C *source);
int8_t SynchClockMsg_cpp_isSubscribedTo(SynchClockMsg_C *subscriber, void* source);

void SynchClockMsg_C_addAuthor(SynchClockMsg_C *coowner, SynchClockMsg_C *data);

void SynchClockMsg_C_init(SynchClockMsg_C *owner);

int SynchClockMsg_C_isLinked(SynchClockMsg_C *data);

int SynchClockMsg_C_isWritten(SynchClockMsg_C *data);

uint64_t SynchClockMsg_C_timeWritten(SynchClockMsg_C *data);

int64_t SynchClockMsg_C_moduleID(SynchClockMsg_C *data);

void SynchClockMsg_C_write(SynchClockMsgPayload *data, SynchClockMsg_C *destination, int64_t moduleID, uint64_t callTime);

SynchClockMsgPayload SynchClockMsg_C_read(SynchClockMsg_C *source);

SynchClockMsgPayload SynchClockMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif