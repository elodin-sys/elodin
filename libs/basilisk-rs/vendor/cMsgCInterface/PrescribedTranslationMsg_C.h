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

#ifndef PrescribedTranslationMsg_C_H
#define PrescribedTranslationMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/PrescribedTranslationMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    PrescribedTranslationMsgPayload payload;		        //!< message copy, zero'd on construction
    PrescribedTranslationMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} PrescribedTranslationMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void PrescribedTranslationMsg_cpp_subscribe(PrescribedTranslationMsg_C *subscriber, void* source);

void PrescribedTranslationMsg_C_subscribe(PrescribedTranslationMsg_C *subscriber, PrescribedTranslationMsg_C *source);

int8_t PrescribedTranslationMsg_C_isSubscribedTo(PrescribedTranslationMsg_C *subscriber, PrescribedTranslationMsg_C *source);
int8_t PrescribedTranslationMsg_cpp_isSubscribedTo(PrescribedTranslationMsg_C *subscriber, void* source);

void PrescribedTranslationMsg_C_addAuthor(PrescribedTranslationMsg_C *coowner, PrescribedTranslationMsg_C *data);

void PrescribedTranslationMsg_C_init(PrescribedTranslationMsg_C *owner);

int PrescribedTranslationMsg_C_isLinked(PrescribedTranslationMsg_C *data);

int PrescribedTranslationMsg_C_isWritten(PrescribedTranslationMsg_C *data);

uint64_t PrescribedTranslationMsg_C_timeWritten(PrescribedTranslationMsg_C *data);

int64_t PrescribedTranslationMsg_C_moduleID(PrescribedTranslationMsg_C *data);

void PrescribedTranslationMsg_C_write(PrescribedTranslationMsgPayload *data, PrescribedTranslationMsg_C *destination, int64_t moduleID, uint64_t callTime);

PrescribedTranslationMsgPayload PrescribedTranslationMsg_C_read(PrescribedTranslationMsg_C *source);

PrescribedTranslationMsgPayload PrescribedTranslationMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif