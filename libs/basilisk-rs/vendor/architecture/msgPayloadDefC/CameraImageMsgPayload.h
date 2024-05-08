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

#ifndef IMAGE_MSG_H
#define IMAGE_MSG_H

#define MAX_FILENAME_LENGTH 10000

/*! @brief Structure used to define the image */

#include "architecture/utilities/macroDefinitions.h"

/*! @brief Structure used to define the output definition for attitude guidance*/
typedef struct {
    uint64_t timeTag;         //!< --[ns]   Current vehicle time-tag associated with measurements*/
    int valid;          //!< --  A valid image is present for 1, 0 if not*/
    int64_t cameraID;          //!< -- [-]   ID of the camera that took the snapshot*/
    void* imagePointer;        //!< -- Pointer to the image
    int32_t imageBufferLength; //!< -- Length of the buffer for recasting
    int8_t imageType;         //!< -- Number of channels in each pixel, RGB = 3, RGBA = 4
}CameraImageMsgPayload;


#endif
