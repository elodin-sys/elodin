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

#ifndef NAV_TRANS_MESSAGE_H
#define NAV_TRANS_MESSAGE_H


/*! @brief Structure used to define the output definition for translatoin guidance*/
typedef struct {
    double timeTag;          //!< [s]   Current vehicle time-tag associated with measurements*/
    double r_BN_N[3];        //!< [m]   Current inertial spacecraft position vector in inertial frame N components
    double v_BN_N[3];        //!< [m/s] Current inertial velocity of the spacecraft in inertial frame N components
    double vehAccumDV[3];    //!< [m/s] Total accumulated delta-velocity for s/c
}NavTransMsgPayload;


#endif
