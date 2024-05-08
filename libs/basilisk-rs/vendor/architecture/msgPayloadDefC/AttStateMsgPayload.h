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

#ifndef ATT_STATE_MESSAGE_H
#define ATT_STATE_MESSAGE_H



/*! @brief Structure used to define the output euler set for attitude reference generation */
typedef struct {
    double state[3];          //!< []   3D attitude orientation coordinate set The units depend on the attitude coordinate chosen and can be either radians (i.e. Euler angles) or dimensionless (i.e. MRP, quaternions, etc.)
    double rate[3];           //!< []   3D attitude rate coordinate set.  These rate coordinates can be either omega (in rad/sec) or attitude coordiante rates with appropriate units
}AttStateMsgPayload;


#endif
