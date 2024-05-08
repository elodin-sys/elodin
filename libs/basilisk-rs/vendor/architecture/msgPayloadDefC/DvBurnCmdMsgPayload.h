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
#include <stdint.h>

#ifndef DV_BURN_CMD_MESSAGE_H
#define DV_BURN_CMD_MESSAGE_H



/*! @brief Input burn command structure used to configure the burn*/
typedef struct {
    double dvInrtlCmd[3];    //!< [m/s] The commanded DV we need in inertial 
    double dvRotVecUnit[3];  //!< [-] The commanded vector we need to rotate about
    double dvRotVecMag;      //!< [r/s] The commanded rotation rate for the vector
    uint64_t burnStartTime;  //!< [ns]  The commanded time to start the burn
}DvBurnCmdMsgPayload;


#endif
