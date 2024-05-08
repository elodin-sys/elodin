/*
 ISC License

 Copyright (c) 2023 Laboratory for Atmospheric and Space Physics, University of Colorado at Boulder

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

#ifndef _CM_EST_DATA_MESSAGE_H
#define _CM_EST_DATA_MESSAGE_H

/*! @brief Structure used to define CM estimation output data */
typedef struct {
    double attError;             //!< [-]   attitude convergence error
    double state[3];             //!< [m]   estimated cm location
    double stateError[3];        //!< [m]   errpr w.r.t. truth
    double covariance[3];        //!< [m^2] CM estimated covariance
    double preFitRes[3];         //!< [Nm]  pre-fit torque residuals
    double postFitRes[3];        //!< [Nm]  post-fit torque residuals
}CMEstDataMsgPayload;



#endif
