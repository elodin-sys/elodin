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

#ifndef _IMAGE_PROC_HOUGH_H_
#define _IMAGE_PROC_HOUGH_H_

#include <stdint.h>
#include <Eigen/Dense>
#include "architecture/messaging/messaging.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"

#include "architecture/msgPayloadDefC/CameraImageMsgPayload.h"
#include "architecture/msgPayloadDefC/OpNavCirclesMsgPayload.h"

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/utilities/avsEigenMRP.h"
#include "architecture/utilities/bskLogging.h"


/*! @brief visual planet tracking with Hough circles */
class HoughCircles: public SysModel {
public:
    HoughCircles();
    ~HoughCircles();
    
    void UpdateState(uint64_t CurrentSimNanos);
    void Reset(uint64_t CurrentSimNanos);
    
public:
    std::string filename;                //!< Filename for module to read an image directly
    Message<OpNavCirclesMsgPayload> opnavCirclesOutMsg;  //!< The name of the OpNavCirclesMsg output message
    ReadFunctor<CameraImageMsgPayload> imageInMsg;          //!< The name of the camera output message
    std::string saveDir;                //!< The name of the directory to save images
    uint64_t sensorTimeTag;              //!< [ns] Current time tag for sensor out
    /* OpenCV specific arguments needed for HoughCircle finding*/
    int32_t blurrSize;                   //!< [px] Size of the blurring box in pixels
    int32_t cannyThresh;                 //!< [px] Canny edge detection Threshold
    int32_t voteThresh;                  //!< [-] Threshold in number of votes to qualify a circle as detected
    int32_t houghMinDist;                //!< [px] Min distance between 2 detected circles
    int32_t houghMinRadius;              //!< [-] Min radius of a detected circle
    int32_t houghMaxRadius;              //!< [-] Max radius of a detected circle
    int32_t dpValue;                     //!< [-] Subscaling of image for circle searching, 1 searches full image
    double noiseSF;                      //!< [-] Scale Factor for noise control
    int32_t expectedCircles;             //!< [-] Number of expected circles to be found
    int32_t saveImages;                  //!< [-] 1 to save images to file for debugging
    BSKLogger bskLogger;                //!< -- BSK Logging
};


#endif

