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

#ifndef ATT_GUID_MESSAGE_H
#define ATT_GUID_MESSAGE_H



/*! @brief Structure used to define the output definition for attitude guidance*/
typedef struct {
    double sigma_BR[3];         //!<        Current attitude error estimate (MRPs) of B relative to R*/
    double omega_BR_B[3];       //!< [r/s]  Current body error estimate of B relateive to R in B frame compoonents */
    double omega_RN_B[3];       //!< [r/s]  Reference frame rate vector of the of R relative to N in B frame components */
    double domega_RN_B[3];      //!< [r/s2] Reference frame inertial body acceleration of R relative to N in B frame components */
}AttGuidMsgPayload;


#endif
