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

#ifndef HEADING_FILTER_MESSAGE_H
#define HEADING_FILTER_MESSAGE_H

#define HEAD_N_STATES 3
#define HEAD_N_STATES_SWITCH 5
#define OPNAV_MEAS 3

/*! @brief structure for filter-states output for the unscented kalman filter
 implementation of the sunline state estimator*/
typedef struct {
    double timeTag;                             /*!< [s] Current time of validity for output */
    double covar[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];    /*!< [-] Current covariance of the filter */
    double state[HEAD_N_STATES_SWITCH];                 /*!< [-] Current estimated state of the filter */
    double stateError[HEAD_N_STATES_SWITCH];                 /*!< [-] Current deviation of the state from the reference state */
    double postFitRes[3];                 /*!< [-] PostFit Residuals  */

}HeadingFilterMsgPayload;



#endif
