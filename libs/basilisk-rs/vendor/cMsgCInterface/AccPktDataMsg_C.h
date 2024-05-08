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

#ifndef AccPktDataMsg_C_H
#define AccPktDataMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AccPktDataMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AccPktDataMsgPayload payload;		        //!< message copy, zero'd on construction
    AccPktDataMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AccPktDataMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AccPktDataMsg_cpp_subscribe(AccPktDataMsg_C *subscriber, void* source);

void AccPktDataMsg_C_subscribe(AccPktDataMsg_C *subscriber, AccPktDataMsg_C *source);

int8_t AccPktDataMsg_C_isSubscribedTo(AccPktDataMsg_C *subscriber, AccPktDataMsg_C *source);
int8_t AccPktDataMsg_cpp_isSubscribedTo(AccPktDataMsg_C *subscriber, void* source);

void AccPktDataMsg_C_addAuthor(AccPktDataMsg_C *coowner, AccPktDataMsg_C *data);

void AccPktDataMsg_C_init(AccPktDataMsg_C *owner);

int AccPktDataMsg_C_isLinked(AccPktDataMsg_C *data);

int AccPktDataMsg_C_isWritten(AccPktDataMsg_C *data);

uint64_t AccPktDataMsg_C_timeWritten(AccPktDataMsg_C *data);

int64_t AccPktDataMsg_C_moduleID(AccPktDataMsg_C *data);

void AccPktDataMsg_C_write(AccPktDataMsgPayload *data, AccPktDataMsg_C *destination, int64_t moduleID, uint64_t callTime);

AccPktDataMsgPayload AccPktDataMsg_C_read(AccPktDataMsg_C *source);

AccPktDataMsgPayload AccPktDataMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif