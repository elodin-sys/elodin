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

#ifndef EPHEMERIS_OUTPUT_H
#define EPHEMERIS_OUTPUT_H


/*! @brief Message structure used to write ephemeris states out to other modules*/
typedef struct {
    double r_BdyZero_N[3];          //!< [m] Position of orbital body
    double v_BdyZero_N[3];          //!< [m/s] Velocity of orbital body
    double sigma_BN[3];             //!< MRP attitude of the orbital body fixed frame relative to inertial
    double omega_BN_B[3];           //!< [r/s] angular velocity of the orbital body relative to inertial
    double timeTag;                 //!< [s] vehicle Time-tag for state
}EphemerisMsgPayload;


#endif
