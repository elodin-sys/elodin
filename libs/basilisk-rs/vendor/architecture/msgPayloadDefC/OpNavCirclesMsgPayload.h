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

#ifndef CIRCLE_OPNAV_MSG_H
#define CIRCLE_OPNAV_MSG_H

/*! @brief Structure used to define circles processed from image*/

#include "architecture/utilities/macroDefinitions.h"

typedef struct {
    uint64_t timeTag;         //!< --[ns]   Current vehicle time-tag associated with measurements
    int valid; //!< --  Valid measurement if 1, not if 0
    int64_t cameraID;          //!< -- [-]   ID of the camera that took the snapshot
    double planetIds[MAX_CIRCLE_NUM];          //!< -- [-]   Ids for identified celestial bodies
    double circlesCenters[2*MAX_CIRCLE_NUM];          //!< -- [-]   Center x, y in pixels of the circles
    double circlesRadii[MAX_CIRCLE_NUM];          //!< -- [-]   Radius rho in pixels of the circles
    double uncertainty[3*3]; //!< -- [-] Uncertainty about the image processing results for x, y, rho (center and radius) for main circle
}OpNavCirclesMsgPayload;


#endif
