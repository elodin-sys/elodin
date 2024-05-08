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

#ifndef RW_SPEED_MESSAGE_STRUCT_H
#define RW_SPEED_MESSAGE_STRUCT_H



#include "architecture/utilities/macroDefinitions.h"


/*! @brief Structure used to define the output definition for reaction wheel speeds*/
typedef struct {
    double wheelSpeeds[MAX_EFF_CNT];                //!< r/s The current angular velocities of the RW wheel
    double wheelThetas[MAX_EFF_CNT];                //!< rad The current angle of the RW if jitter is enabled
}RWSpeedMsgPayload;


#endif
