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

#ifndef _OPNAV_POINT_H_
#define _OPNAV_POINT_H_


#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CameraConfigMsg_C.h"
#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/OpNavMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>

/*! @brief module configuration message definition
 */
typedef struct {
    AttGuidMsg_C attGuidanceOutMsg;                 /*!< The name of the output message*/
    OpNavMsg_C opnavDataInMsg;                      /*!< The name of the Input message*/
    NavAttMsg_C imuInMsg;                           /*!< The name of the incoming IMU message*/
    CameraConfigMsg_C cameraConfigInMsg;            //!< The name of the camera config message

    double minUnitMag;       /*!< -- The minimally acceptable norm of opNav body vector*/
    double opNavAngleErr;      /*!< -- rad The current error between cmd and obs opNav angle*/
    double smallAngle;       /*!< -- rad An angle value that specifies what is near 0 or 180 degrees */
    double eHat180_B[3];     /*!< -- Eigen axis to use if commanded axis is 180 from opNav axis */
    double opNavMnvrVec[3];    /*!< -- The eigen axis that we want to rotate on to see target*/
    double lastTime; /*!< -- Last time a measurement came in to integrate pointing */
    double timeOut;  /*!< -- If no images were seen in this much time, stop using past values */
    double alignAxis_C[3];    /*!< -- Desired camera vector to point at target*/
    double currentHeading_N[3];   /*!< -- Previous heading command in intertial Frame*/
    double omega_RN_B[3];    /*!< -- Desired body rate vector if no opNav direction is available */
    double opNavAxisSpinRate;  /*!< -- r/s Desired constant spin rate about opNav vector */

    AttGuidMsgPayload attGuidanceOutBuffer;   /*!< -- The output data that we compute*/
    BSKLogger *bskLogger;                             //!< BSK Logging
}OpNavPointConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_opNavPoint(OpNavPointConfig *configData, int64_t moduleID);
    void Update_opNavPoint(OpNavPointConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_opNavPoint(OpNavPointConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
