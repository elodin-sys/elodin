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

#ifndef DvBurnCmdMsg_C_H
#define DvBurnCmdMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/DvBurnCmdMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    DvBurnCmdMsgPayload payload;		        //!< message copy, zero'd on construction
    DvBurnCmdMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} DvBurnCmdMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void DvBurnCmdMsg_cpp_subscribe(DvBurnCmdMsg_C *subscriber, void* source);

void DvBurnCmdMsg_C_subscribe(DvBurnCmdMsg_C *subscriber, DvBurnCmdMsg_C *source);

int8_t DvBurnCmdMsg_C_isSubscribedTo(DvBurnCmdMsg_C *subscriber, DvBurnCmdMsg_C *source);
int8_t DvBurnCmdMsg_cpp_isSubscribedTo(DvBurnCmdMsg_C *subscriber, void* source);

void DvBurnCmdMsg_C_addAuthor(DvBurnCmdMsg_C *coowner, DvBurnCmdMsg_C *data);

void DvBurnCmdMsg_C_init(DvBurnCmdMsg_C *owner);

int DvBurnCmdMsg_C_isLinked(DvBurnCmdMsg_C *data);

int DvBurnCmdMsg_C_isWritten(DvBurnCmdMsg_C *data);

uint64_t DvBurnCmdMsg_C_timeWritten(DvBurnCmdMsg_C *data);

int64_t DvBurnCmdMsg_C_moduleID(DvBurnCmdMsg_C *data);

void DvBurnCmdMsg_C_write(DvBurnCmdMsgPayload *data, DvBurnCmdMsg_C *destination, int64_t moduleID, uint64_t callTime);

DvBurnCmdMsgPayload DvBurnCmdMsg_C_read(DvBurnCmdMsg_C *source);

DvBurnCmdMsgPayload DvBurnCmdMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif