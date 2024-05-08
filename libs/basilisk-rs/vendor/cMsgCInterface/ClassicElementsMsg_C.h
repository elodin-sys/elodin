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

#ifndef ClassicElementsMsg_C_H
#define ClassicElementsMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/ClassicElementsMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    ClassicElementsMsgPayload payload;		        //!< message copy, zero'd on construction
    ClassicElementsMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} ClassicElementsMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void ClassicElementsMsg_cpp_subscribe(ClassicElementsMsg_C *subscriber, void* source);

void ClassicElementsMsg_C_subscribe(ClassicElementsMsg_C *subscriber, ClassicElementsMsg_C *source);

int8_t ClassicElementsMsg_C_isSubscribedTo(ClassicElementsMsg_C *subscriber, ClassicElementsMsg_C *source);
int8_t ClassicElementsMsg_cpp_isSubscribedTo(ClassicElementsMsg_C *subscriber, void* source);

void ClassicElementsMsg_C_addAuthor(ClassicElementsMsg_C *coowner, ClassicElementsMsg_C *data);

void ClassicElementsMsg_C_init(ClassicElementsMsg_C *owner);

int ClassicElementsMsg_C_isLinked(ClassicElementsMsg_C *data);

int ClassicElementsMsg_C_isWritten(ClassicElementsMsg_C *data);

uint64_t ClassicElementsMsg_C_timeWritten(ClassicElementsMsg_C *data);

int64_t ClassicElementsMsg_C_moduleID(ClassicElementsMsg_C *data);

void ClassicElementsMsg_C_write(ClassicElementsMsgPayload *data, ClassicElementsMsg_C *destination, int64_t moduleID, uint64_t callTime);

ClassicElementsMsgPayload ClassicElementsMsg_C_read(ClassicElementsMsg_C *source);

ClassicElementsMsgPayload ClassicElementsMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif