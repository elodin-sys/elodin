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

#ifndef CAMERA_MSG_H
#define CAMERA_MSG_H

#define MAX_STRING_LENGTH 256
/*! @brief Structure used to define the camera parameters*/

#include "architecture/utilities/macroDefinitions.h"

typedef struct {
    int64_t cameraID;          //!< [-]   ID of the camera that took the snapshot*/
    int isOn; //!<  The camera is taking images at rendering rate if 1, 0 if not*/
    char parentName[MAX_STRING_LENGTH];  //!< [-] Name of the parent body to which the camera should be attached
    double fieldOfView;        //!< [rad]   Camera Field of View, edge-to-edge along camera y-axis */
    int resolution[2];         //!< [-] Camera resolution, width/height in pixels (pixelWidth/pixelHeight in Unity) in pixels*/
    uint64_t renderRate;       //!< [ns] Frame time interval at which to capture images in units of nanosecond */
    double cameraPos_B[3];     //!< [m] Camera position in body frame */
    double sigma_CB[3];        //!< [-] MRP defining the orientation of the camera frame relative to the body frame */
    char skyBox[MAX_STRING_LENGTH]; //!< string containing the star field preference
    int postProcessingOn;       //!< (Optional) Enable post-processing of camera image. Value of 0 (protobuffer default) to use viz default which is off, -1 for false, 1 for true
    double ppFocusDistance;     //!< (Optional) Distance to the point of focus, minimum value of 0.1, Value of 0 to turn off this parameter entirely.
    double ppAperture;          //!<  (Optional) Ratio of the aperture (known as f-stop or f-number). The smaller the value is, the shallower the depth of field is. Valid Setting Range: 0.05 to 32. Value of 0 to turn off this parameter entirely.
    double ppFocalLength;       //!< [m] (Optional) Valid setting range: 0.001m to 0.3m. Value of 0 to turn off this parameter entirely.
    int ppMaxBlurSize;          //!< (Optional) Convolution kernel size of the bokeh filter, which determines the maximum radius of bokeh. It also affects the performance (the larger the kernel is, the longer the GPU time is required). Depth textures Value of 1 for Small, 2 for Medium, 3 for Large, 4 for Extra Large. Value of 0 to turn off this parameter entirely.
    int updateCameraParameters; //!< If true, commands camera to update Instrument Camera to current message's parameters
    int renderMode; //!< (Optional) Value of 0 to render visual image (default), value of 1 to render depth buffer to image
    double depthMapClippingPlanes[2]; //!< (Optional) [m] Set the bounds of rendered depth map by setting the near and far clipping planes when in renderMode=1 (depthMap mode). Default values of 0.1 and 100.
}CameraConfigMsgPayload;

#endif
