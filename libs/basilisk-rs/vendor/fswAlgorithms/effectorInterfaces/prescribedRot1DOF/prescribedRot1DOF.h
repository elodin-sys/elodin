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

#ifndef _PRESCRIBEDROT1DOF_
#define _PRESCRIBEDROT1DOF_

#include <stdint.h>
#include <stdbool.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/HingedRigidBodyMsg_C.h"
#include "cMsgCInterface/PrescribedRotationMsg_C.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* User configurable variables */
    double thetaDDotMax;                                        //!< [rad/s^2] Maximum angular acceleration of spinning body
    double rotAxis_M[3];                                        //!< Rotation axis for the maneuver in M frame components
    double omega_FM_F[3];                                       //!< [rad/s] Angular velocity of frame F wrt frame M in F frame components
    double omegaPrime_FM_F[3];                                  //!< [rad/s^2] B frame time derivative of omega_FM_F in F frame components
    double sigma_FM[3];                                         //!< MRP attitude of frame F with respect to frame M

    /* Private variables */
    bool convergence;                                           //!< Boolean variable is true when the maneuver is complete
    double tInit;                                               //!< [s] Simulation time at the beginning of the maneuver
    double thetaInit;                                           //!< [rad] Initial spinning body angle from frame M to frame F about rotAxis_M
    double thetaDotInit;                                        //!< [rad/s] Initial spinning body angle rate between frame M to frame F
    double thetaRef;                                            //!< [rad] Reference angle from frame M to frame F about rotAxis_M
    double thetaDotRef;                                         //!< [rad/s] Reference angle rate between frame M to frame F
    double ts;                                                  //!< [s] The simulation time halfway through the maneuver (switch time for ang accel)
    double tf;                                                  //!< [s] Simulation time when the maneuver is finished
    double a;                                                   //!< Parabolic constant for the first half of the maneuver
    double b;                                                   //!< Parabolic constant for the second half of the maneuver

    BSKLogger *bskLogger;                                       //!< BSK Logging

    /* Messages */
    HingedRigidBodyMsg_C spinningBodyInMsg;                     //!< Input msg for the spinning body reference angle and angle rate
    HingedRigidBodyMsg_C spinningBodyOutMsg;                    //!< Output msg for the spinning body angle and angle rate
    PrescribedRotationMsg_C prescribedRotationOutMsg;           //!< Output msg for the spinning body prescribed rotational states

}PrescribedRot1DOFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, int64_t moduleID);                     //!< Method for module initialization
    void Reset_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, uint64_t callTime, int64_t moduleID);     //!< Method for module reset
    void Update_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, uint64_t callTime, int64_t moduleID);    //!< Method for module time update
#ifdef __cplusplus
}
#endif

#endif
