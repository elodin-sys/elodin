/*
 ISC License
 
 Copyright (c) 2023, Laboratory  for Atmospheric and Space Physics, University of Colorado at Boulder
 
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


/*! @brief Top level structure for the thrust CM estimation kalman filter.
 Used to estimate the spacecraft's center of mass position with respect to the B frame.
 */

#ifndef THRUSTCMESTIMATION_H
#define THRUSTCMESTIMATION_H

#include "architecture/msgPayloadDefC/AttGuidMsgPayload.h"
#include "architecture/msgPayloadDefC/CmdTorqueBodyMsgPayload.h"
#include "architecture/msgPayloadDefC/CMEstDataMsgPayload.h"
#include "architecture/msgPayloadDefC/THRConfigMsgPayload.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include <string.h>
#include <array>
#include <math.h>

class ThrustCMEstimation: public SysModel {
public:
    ThrustCMEstimation();
    ~ThrustCMEstimation() override;
    void SelfInit() override;
    void Reset(uint64_t CurrentSimNanos) override;
    void UpdateState(uint64_t CurrentSimNanos) override;

    /*! declare these user-defined quantities */
    double attitudeTol;

    ReadFunctor<THRConfigMsgPayload>        thrusterConfigBInMsg;     //!< thr config in msg in B-frame coordinates
    ReadFunctor<CmdTorqueBodyMsgPayload>    intFeedbackTorqueInMsg;   //!< integral feedback torque input msg
    ReadFunctor<AttGuidMsgPayload>          attGuidInMsg;             //!< attitude guidance input msg
    ReadFunctor<VehicleConfigMsgPayload>    vehConfigInMsg;           //!< (optional) vehicle configuration input msg
    Message<CMEstDataMsgPayload>            cmEstDataOutMsg;          //!< estimated CM output msg
    Message<VehicleConfigMsgPayload>        vehConfigOutMsg;          //!< output C++ vehicle configuration msg
    VehicleConfigMsg_C                      vehConfigOutMsgC = {};    //!< output C vehicle configuration msg

    Eigen::Vector3d r_CB_B;                 //!< initial CM estimate
    Eigen::Vector3d P0;                     //!< initial CM state covariance
    Eigen::Vector3d R0;                     //!< measurement noise covariance

private:
    Eigen::Matrix3d I;                      //!< identity matrix
    Eigen::Matrix3d P;                      //!< state covariance
    Eigen::Matrix3d R;                      //!< measurement noise covariance
    Eigen::Vector3d r_CB_est;               //!< CM location estimate

    bool cmKnowledge;                       //!< boolean to assess if vehConfigInMsg is connected

    BSKLogger bskLogger; //!< -- BSK Logging
};

#endif
