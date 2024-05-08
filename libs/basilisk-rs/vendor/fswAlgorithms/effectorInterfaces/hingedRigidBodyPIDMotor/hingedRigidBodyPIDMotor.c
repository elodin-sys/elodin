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


/* modify the path to reflect the new module names */
#include "hingedRigidBodyPIDMotor.h"
#include "string.h"
#include <math.h>

/* Support files */
#include "architecture/utilities/macroDefinitions.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_hingedRigidBodyPIDMotor(hingedRigidBodyPIDMotorConfig *configData, int64_t moduleID)
{
    ArrayMotorTorqueMsg_C_init(&configData->motorTorqueOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_hingedRigidBodyPIDMotor(hingedRigidBodyPIDMotorConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!HingedRigidBodyMsg_C_isLinked(&configData->hingedRigidBodyInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: solarArrayAngle.hingedRigidBodyInMsg wasn't connected.");
    }
    if (!HingedRigidBodyMsg_C_isLinked(&configData->hingedRigidBodyRefInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: solarArrayAngle.hingedRigidBodyRefInMsg wasn't connected.");
    }

    /*! initialize module parameters to compute integral error via trapezoid integration */
    configData->priorTime = 0;
    configData->priorThetaError = 0;
    configData->intError = 0;
}

/*! This method computes the control torque to the solar array drive based on a PD control law
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_hingedRigidBodyPIDMotor(hingedRigidBodyPIDMotorConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Create and assign buffer messages */
    ArrayMotorTorqueMsgPayload motorTorqueOut = ArrayMotorTorqueMsg_C_zeroMsgPayload();
    HingedRigidBodyMsgPayload  hingedRigidBodyIn = HingedRigidBodyMsg_C_read(&configData->hingedRigidBodyInMsg);
    HingedRigidBodyMsgPayload  hingedRigidBodyRefIn = HingedRigidBodyMsg_C_read(&configData->hingedRigidBodyRefInMsg);

    /*! compute angle error and error rate */
    double thetaError    = hingedRigidBodyRefIn.theta - hingedRigidBodyIn.theta;
    double thetaErrorDot = hingedRigidBodyRefIn.thetaDot - hingedRigidBodyIn.thetaDot;

    /*! extract gains from input */
    double K = configData->K;
    double P = configData->P;
    double I = configData->I;

    /*! compute integral term */
    double dt;
    if (callTime != 0) {
        dt = (callTime - configData->priorTime) * NANO2SEC;
        configData->intError += (thetaError + configData->priorThetaError) * dt / 2;
    }

    /*! update stored quantities */
    configData->priorThetaError = thetaError;
    configData->priorTime = callTime;

    /*! compute torque */
    double T = K * thetaError + P * thetaErrorDot + I * configData->intError;
    motorTorqueOut.motorTorque[0] = T;

    /*! write output message */
    ArrayMotorTorqueMsg_C_write(&motorTorqueOut, &configData->motorTorqueOutMsg, moduleID, callTime);
}
