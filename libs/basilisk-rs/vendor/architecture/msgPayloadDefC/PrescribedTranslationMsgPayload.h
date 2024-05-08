/*
 ISC License

 Copyright (c) 2024, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef prescribedTranslationSimMsg_h
#define prescribedTranslationSimMsg_h


 /*! @brief Structure used to define the prescribed motion state effector translational state data message */
typedef struct {
    double r_FM_M[3];                          //!< [m] Position vector from the M frame origin to the F frame origin expressed in M frame components
    double rPrime_FM_M[3];                     //!< [m/s] B/M frame time derivative of r_FM_M
    double rPrimePrime_FM_M[3];                //!< [m/s^2] B/M frame time derivative of rPrime_FM_M
}PrescribedTranslationMsgPayload;


#endif /* prescribedTranslationSimMsg_h */
