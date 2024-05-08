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

#ifndef EphemerisMsg_C_H
#define EphemerisMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/EphemerisMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    EphemerisMsgPayload payload;		        //!< message copy, zero'd on construction
    EphemerisMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} EphemerisMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void EphemerisMsg_cpp_subscribe(EphemerisMsg_C *subscriber, void* source);

void EphemerisMsg_C_subscribe(EphemerisMsg_C *subscriber, EphemerisMsg_C *source);

int8_t EphemerisMsg_C_isSubscribedTo(EphemerisMsg_C *subscriber, EphemerisMsg_C *source);
int8_t EphemerisMsg_cpp_isSubscribedTo(EphemerisMsg_C *subscriber, void* source);

void EphemerisMsg_C_addAuthor(EphemerisMsg_C *coowner, EphemerisMsg_C *data);

void EphemerisMsg_C_init(EphemerisMsg_C *owner);

int EphemerisMsg_C_isLinked(EphemerisMsg_C *data);

int EphemerisMsg_C_isWritten(EphemerisMsg_C *data);

uint64_t EphemerisMsg_C_timeWritten(EphemerisMsg_C *data);

int64_t EphemerisMsg_C_moduleID(EphemerisMsg_C *data);

void EphemerisMsg_C_write(EphemerisMsgPayload *data, EphemerisMsg_C *destination, int64_t moduleID, uint64_t callTime);

EphemerisMsgPayload EphemerisMsg_C_read(EphemerisMsg_C *source);

EphemerisMsgPayload EphemerisMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif