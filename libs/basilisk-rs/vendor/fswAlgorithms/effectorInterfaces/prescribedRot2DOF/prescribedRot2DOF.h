/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef _PRESCRIBEDROT2DOF_
#define _PRESCRIBEDROT2DOF_

/*! Include the required files. */
#include <stdint.h>
#include <stdbool.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/HingedRigidBodyMsg_C.h"
#include "cMsgCInterface/PrescribedRotationMsg_C.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct
{
    /* User configurable variables */
    double phiDDotMax;                                         //!< [rad/s^2] Maximum angular acceleration of the spinning body
    double rotAxis1_M[3];                                      //!< M frame rotation axis for the first rotation
    double rotAxis2_F1[3];                                     //!< F1 frame intermediate rotation axis for the second rotation

    /* Private variables */
    double omega_FM_F[3];                                      //!< [rad/s] angular velocity of frame F relative to frame M in F frame components
    double omegaPrime_FM_F[3];                                 //!< [rad/s^2] B frame time derivative of omega_FB_F in F frame components
    double sigma_FM[3];                                        //!< MRP attitude of frame F relative to frame M
    bool isManeuverComplete;                                   //!< Boolean variable is true when the attitude maneuver is complete
    double maneuverStartTime;                                  //!< [s] Simulation time at the start of the attitude maneuver
    double rotAxis_M[3];                                       //!< Reference PRV axis expressed in M frame components
    double phiRef;                                             //!< [rad] Reference PRV angle (The positive short rotation is chosen)
    double phiDotRef;                                          //!< [rad/s] Reference PRV angle rate
    double phi;                                                //!< [rad] Current PRV angle
    double phiRefAccum;                                        //!< [rad] This variable logs the accumulated reference PRV angles
    double phiAccum;                                           //!< [rad] This variable logs the accumulated current PRV angle
    double maneuverSwitchTime;                                 //!< [s] Simulation time halfway through the attitude maneuver (switch time)
    double maneuverEndTime;                                    //!< [s] Simulation time when the maneuver is complete
    double a;                                                  //!< Parabolic constant for the first half of the attitude maneuver
    double b;                                                  //!< Parabolic constant for the second half of the attitude maneuver
    double dcm_F0M[3][3];                                      //!< DCM from the M frame to the spinning body body frame at the beginning of the maneuver

    /* Declare the module input-output messages */
    HingedRigidBodyMsg_C spinningBodyRef1InMsg;                //!< Input msg for the first reference angle and angle rate
    HingedRigidBodyMsg_C spinningBodyRef2InMsg;                //!< Input msg for the second reference angles and angle rate
    PrescribedRotationMsg_C prescribedRotationOutMsg;          //!< Output msg for the profiled prescribed rotational states

    BSKLogger *bskLogger;                                      //!< BSK Logging

}PrescribedRot2DOFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, int64_t moduleID);                         //<! Method for initializing the module
    void Reset_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, uint64_t callTime, int64_t moduleID);         //<! Method for resetting the module
    void Update_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, uint64_t callTime, int64_t moduleID);        //<! Method for the updating the module
#ifdef __cplusplus
}
#endif

#endif
