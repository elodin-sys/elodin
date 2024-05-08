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

#ifndef NAV_ATT_MESSAGE_H
#define NAV_ATT_MESSAGE_H

/*! @brief Structure used to define the output definition for attitude guidance*/
typedef struct {
    double timeTag;          //!< [s]   Current vehicle time-tag associated with measurements*/
    double sigma_BN[3];      //!<       Current spacecraft attitude (MRPs) of body relative to inertial */
    double omega_BN_B[3];    //!< [r/s] Current spacecraft angular velocity vector of body frame B relative to inertial frame N, in B frame components
    double vehSunPntBdy[3];  //!<       Current sun pointing vector in body frame
}NavAttMsgPayload;


#endif
