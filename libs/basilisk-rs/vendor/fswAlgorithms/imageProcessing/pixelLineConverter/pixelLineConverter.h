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

#ifndef _PIXEL_LINE_CONVERTER_H_
#define _PIXEL_LINE_CONVERTER_H_

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/OpNavCirclesMsg_C.h"
#include "cMsgCInterface/CameraConfigMsg_C.h"
#include "cMsgCInterface/OpNavMsg_C.h"

#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/bskLogging.h"


/*! @brief The configuration structure for the pixelLine Converter module.*/
typedef struct {
    OpNavMsg_C opNavOutMsg; //!< [-] output navigation message for relative position
    CameraConfigMsg_C cameraConfigInMsg; //!< camera config input message
    NavAttMsg_C attInMsg; //!< attitude input message
    OpNavCirclesMsg_C circlesInMsg; //!< circles input message

    int32_t planetTarget; //!< The planet targeted (None = 0, Earth = 1, Mars = 2, Jupiter = 3 are allowed)

    BSKLogger *bskLogger;                             //!< BSK Logging
}PixelLineConvertData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_pixelLineConverter(PixelLineConvertData *configData, int64_t moduleID);
    void Update_pixelLineConverter(PixelLineConvertData *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_pixelLineConverter(PixelLineConvertData *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
