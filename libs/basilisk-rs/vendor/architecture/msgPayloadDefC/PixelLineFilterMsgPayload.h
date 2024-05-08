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

#ifndef PIXLINE_FILTER_MESSAGE_H
#define PIXLINE_FILTER_MESSAGE_H


#define PIXLINE_N_STATES 9
#define PIXLINE_DYN_STATES 6
#define PIXLINE_N_MEAS 6

/*! @brief structure for filter-states output for the unscented kalman filter
 implementation of the sunline state estimator*/
typedef struct {
    double timeTag;                                     //!< [s] Current time of validity for output
    double covar[PIXLINE_N_STATES*PIXLINE_N_STATES];    //!< [-] Current covariance of the filter
    double state[PIXLINE_N_STATES];                     //!< [-] Current estimated state of the filter
    double stateError[PIXLINE_N_STATES];                //!< [-] Current deviation of the state from the reference state
    double postFitRes[PIXLINE_N_MEAS];                  //!< [-] PostFit Residuals
    int numObs;                                         //!< [-] Valid observation count for this frame
}PixelLineFilterMsgPayload;


#endif
