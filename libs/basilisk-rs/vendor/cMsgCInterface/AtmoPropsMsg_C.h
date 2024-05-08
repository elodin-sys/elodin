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

#ifndef AtmoPropsMsg_C_H
#define AtmoPropsMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/AtmoPropsMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    AtmoPropsMsgPayload payload;		        //!< message copy, zero'd on construction
    AtmoPropsMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} AtmoPropsMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void AtmoPropsMsg_cpp_subscribe(AtmoPropsMsg_C *subscriber, void* source);

void AtmoPropsMsg_C_subscribe(AtmoPropsMsg_C *subscriber, AtmoPropsMsg_C *source);

int8_t AtmoPropsMsg_C_isSubscribedTo(AtmoPropsMsg_C *subscriber, AtmoPropsMsg_C *source);
int8_t AtmoPropsMsg_cpp_isSubscribedTo(AtmoPropsMsg_C *subscriber, void* source);

void AtmoPropsMsg_C_addAuthor(AtmoPropsMsg_C *coowner, AtmoPropsMsg_C *data);

void AtmoPropsMsg_C_init(AtmoPropsMsg_C *owner);

int AtmoPropsMsg_C_isLinked(AtmoPropsMsg_C *data);

int AtmoPropsMsg_C_isWritten(AtmoPropsMsg_C *data);

uint64_t AtmoPropsMsg_C_timeWritten(AtmoPropsMsg_C *data);

int64_t AtmoPropsMsg_C_moduleID(AtmoPropsMsg_C *data);

void AtmoPropsMsg_C_write(AtmoPropsMsgPayload *data, AtmoPropsMsg_C *destination, int64_t moduleID, uint64_t callTime);

AtmoPropsMsgPayload AtmoPropsMsg_C_read(AtmoPropsMsg_C *source);

AtmoPropsMsgPayload AtmoPropsMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif