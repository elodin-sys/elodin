/*
 ISC License

 Copyright (c) 2016-2018, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef LIMB_MSG_H
#define LIMB_MSG_H

#include "architecture/utilities/macroDefinitions.h"


/*! @brief Structure used to define the message containing planet limb data for opNav*/
typedef struct {
    double timeTag;         //!< --[s]   Current vehicle time-tag associated with measurements
    int valid; //!< --  Valid measurement if 1, not if 0
    int32_t numLimbPoints;                      //!< -- [-] Number of limb points found
    int64_t cameraID;          //!< -- [-]   ID of the camera that took the snapshot
    double planetIds;          //!< -- [-]   ID for identified celestial body
    double limbPoints[2*MAX_LIMB_PNTS];          //!< -- [-] (x, y) in pixels of the limb points
}OpNavLimbMsgPayload;


#endif

