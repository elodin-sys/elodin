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

#ifndef _ET_SPHERICAL_CONTROL_H_
#define _ET_SPHERICAL_CONTROL_H_

#include <stdint.h>

#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/CmdForceInertialMsg_C.h"
#include "cMsgCInterface/CmdForceBodyMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/orbitalMotion.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    // declare module IO interfaces
    NavTransMsg_C servicerTransInMsg;                   //!< servicer orbit input message
    NavTransMsg_C debrisTransInMsg;                     //!< debris orbit input message
    NavAttMsg_C servicerAttInMsg;                       //!< servicer attitude input message
    VehicleConfigMsg_C servicerVehicleConfigInMsg;      //!< servicer vehicle configuration (mass) input message
    VehicleConfigMsg_C debrisVehicleConfigInMsg;        //!< debris vehicle configuration (mass) input message
    CmdForceInertialMsg_C eForceInMsg;                  //!< servicer electrostatic force input message
    CmdForceInertialMsg_C forceInertialOutMsg;          //!< servicer inertial frame control force output message
    CmdForceBodyMsg_C forceBodyOutMsg;                  //!< servicer body frame control force output message
    
    double mu;                                          //!< [m^3/s^2]  gravitational parameter
    double L_r;                                         //!< [m]  reference separation distance
    double theta_r;                                     //!< [rad]  reference in-plane rotation angle
    double phi_r;                                       //!< [rad]  reference out-of-plane rotation angle
    double K[9];                                        //!< 3x3 symmetric positive definite feedback gain matrix [K]
    double P[9];                                        //!< 3x3 symmetric positive definite feedback gain matrix [P]
    BSKLogger *bskLogger;                               //!< BSK Logging
} etSphericalControlConfig;

#ifdef __cplusplus
extern "C" {
#endif

void SelfInit_etSphericalControl(etSphericalControlConfig *configData, int64_t moduleID);
void Update_etSphericalControl(etSphericalControlConfig *configData, uint64_t callTime, int64_t moduleID);
void Reset_etSphericalControl(etSphericalControlConfig *configData, uint64_t callTime, int64_t moduleID);
void calc_RelativeMotionControl(etSphericalControlConfig *configData, NavTransMsgPayload servicerTransInMsgBuffer,
                                NavTransMsgPayload debrisTransInMsgBuffer, NavAttMsgPayload servicerAttInMsgBuffer,
                                VehicleConfigMsgPayload servicerVehicleConfigInMsgBuffer,
                                VehicleConfigMsgPayload debrisVehicleConfigInMsgBuffer,
                                CmdForceInertialMsgPayload eForceInMsgBuffer,
                                CmdForceInertialMsgPayload *forceInertialOutMsgBuffer,
                                CmdForceBodyMsgPayload *forceBodyOutMsgBuffer);
#ifdef __cplusplus
}
#endif


#endif
