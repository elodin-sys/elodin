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

#ifndef DvExecutionDataMsg_C_H
#define DvExecutionDataMsg_C_H

#include <stdint.h>
#include "architecture/msgPayloadDefC/DvExecutionDataMsgPayload.h"
#include "architecture/messaging/msgHeader.h"

//! structure definition
typedef struct {
    MsgHeader header;              //!< message header, zero'd on construction
    DvExecutionDataMsgPayload payload;		        //!< message copy, zero'd on construction
    DvExecutionDataMsgPayload *payloadPointer;	    //!< pointer to message
    MsgHeader *headerPointer;      //!< pointer to message header
} DvExecutionDataMsg_C;

#ifdef __cplusplus
extern "C" {
#endif

void DvExecutionDataMsg_cpp_subscribe(DvExecutionDataMsg_C *subscriber, void* source);

void DvExecutionDataMsg_C_subscribe(DvExecutionDataMsg_C *subscriber, DvExecutionDataMsg_C *source);

int8_t DvExecutionDataMsg_C_isSubscribedTo(DvExecutionDataMsg_C *subscriber, DvExecutionDataMsg_C *source);
int8_t DvExecutionDataMsg_cpp_isSubscribedTo(DvExecutionDataMsg_C *subscriber, void* source);

void DvExecutionDataMsg_C_addAuthor(DvExecutionDataMsg_C *coowner, DvExecutionDataMsg_C *data);

void DvExecutionDataMsg_C_init(DvExecutionDataMsg_C *owner);

int DvExecutionDataMsg_C_isLinked(DvExecutionDataMsg_C *data);

int DvExecutionDataMsg_C_isWritten(DvExecutionDataMsg_C *data);

uint64_t DvExecutionDataMsg_C_timeWritten(DvExecutionDataMsg_C *data);

int64_t DvExecutionDataMsg_C_moduleID(DvExecutionDataMsg_C *data);

void DvExecutionDataMsg_C_write(DvExecutionDataMsgPayload *data, DvExecutionDataMsg_C *destination, int64_t moduleID, uint64_t callTime);

DvExecutionDataMsgPayload DvExecutionDataMsg_C_read(DvExecutionDataMsg_C *source);

DvExecutionDataMsgPayload DvExecutionDataMsg_C_zeroMsgPayload();

#ifdef __cplusplus
}
#endif
#endif