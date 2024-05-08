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

#ifndef RECONFIG_BURN_INFO_H
#define RECONFIG_BURN_INFO_H

//! @brief Container for the orbit reconfiguration burn information.
typedef struct {
    int flag;        //!< 0:not scheduled yet, 1:not burned yet, 2:already burned, 3:skipped (combined with another burn)
    double t;            //!< [s] simulation time of the scheduled burn since module reset
    double thrustOnTime; //!< thrust on duration time [s]
    double sigma_RN[3];  //!< target attitude
}ReconfigBurnInfoMsgPayload;


#endif