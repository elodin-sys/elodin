/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef AVS_SMALLBODYNAVMSGPAYLOAD_H
#define AVS_SMALLBODYNAVMSGPAYLOAD_H

#define SMALL_BODY_NAV_N_STATES 12

//! @brief Full states and covariances associated with spacecraft navigation about a small body
typedef struct{
    double state[SMALL_BODY_NAV_N_STATES];  //!< Current state estimate from the filter
    double covar[SMALL_BODY_NAV_N_STATES][SMALL_BODY_NAV_N_STATES];  //!< Current covariance of the filter
}SmallBodyNavMsgPayload;

#endif //AVS_SMALLBODYNAVMSGPAYLOAD_H
