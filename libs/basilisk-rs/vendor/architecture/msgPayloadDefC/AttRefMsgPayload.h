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

#ifndef ATT_REF_MESSAGE_H
#define ATT_REF_MESSAGE_H



/*! @brief Structure used to define the output definition for attitude reference generation */
typedef struct {
    double sigma_RN[3];         //!<        MRP Reference attitude of R relative to N
    double omega_RN_N[3];       //!< [r/s]  Reference frame rate vector of the of R relative to N in N frame components
    double domega_RN_N[3];      //!< [r/s2] Reference frame inertial acceleration of  R relative to N in N frame components
}AttRefMsgPayload;


#endif
