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

/* Import the module header file */
#include "prescribedRot1DOF.h"

/* Other required files to import */
#include <stdbool.h>
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"

/*! This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, int64_t moduleID)
{
    // Initialize the output messages
    PrescribedRotationMsg_C_init(&configData->prescribedRotationOutMsg);
    HingedRigidBodyMsg_C_init(&configData->spinningBodyOutMsg);
}


/*! This method performs a complete reset of the module. The input messages are checked to ensure they are linked.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Reset_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Check if the required input message is linked
    if (!HingedRigidBodyMsg_C_isLinked(&configData->spinningBodyInMsg))
    {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: prescribedRot1DOF.spinningBodyInMsg wasn't connected.");
    }

    // Set the initial time
    configData->tInit = 0.0;

    // Set the initial convergence to true to enter the required loop in Update_prescribedRot1DOF() on the first pass
    configData->convergence = true;
}


/*! This method profiles the prescribed trajectory and updates the prescribed states as a function of time.
The prescribed states are then written to the output message.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Update_prescribedRot1DOF(PrescribedRot1DOFConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Create the buffer messages
    HingedRigidBodyMsgPayload spinningBodyIn;
    HingedRigidBodyMsgPayload spinningBodyOut;
    PrescribedRotationMsgPayload prescribedRotationOut;

    // Zero the output messages
    spinningBodyOut = HingedRigidBodyMsg_C_zeroMsgPayload();
    prescribedRotationOut = PrescribedRotationMsg_C_zeroMsgPayload();

    // Read the input message
    spinningBodyIn = HingedRigidBodyMsg_C_zeroMsgPayload();
    if (HingedRigidBodyMsg_C_isWritten(&configData->spinningBodyInMsg))
    {
        spinningBodyIn = HingedRigidBodyMsg_C_read(&configData->spinningBodyInMsg);
    }

    /* This loop is entered (a) initially and (b) when each attitude maneuver is complete. The reference angle is updated
    even if a new message is not written */
    if (HingedRigidBodyMsg_C_timeWritten(&configData->spinningBodyInMsg) <= callTime && configData->convergence)
    {
        // Store the initial time as the current simulation time
        configData->tInit = callTime * NANO2SEC;

        // Calculate the current ange and angle rate
        double prv_FM_array[3];
        MRP2PRV(configData->sigma_FM, prv_FM_array);
        configData->thetaInit = v3Dot(prv_FM_array, configData->rotAxis_M);
        configData->thetaDotInit = v3Norm(configData->omega_FM_F);

        // Store the reference angle and reference angle rate
        configData->thetaRef = spinningBodyIn.theta;
        configData->thetaDotRef = spinningBodyIn.thetaDot;

        // Define temporal information for the maneuver
        double convTime = sqrt(((0.5 * fabs(configData->thetaRef - configData->thetaInit)) * 8) / configData->thetaDDotMax);
        configData->tf = configData->tInit + convTime;
        configData->ts = configData->tInit + convTime / 2;

        // Define the parabolic constants for the first and second half of the maneuver
        configData->a = 0.5 * (configData->thetaRef - configData->thetaInit) / ((configData->ts - configData->tInit) * (configData->ts - configData->tInit));
        configData->b = -0.5 * (configData->thetaRef - configData->thetaInit) / ((configData->ts - configData->tf) * (configData->ts - configData->tf));

        // Set the convergence to false until the attitude maneuver is complete
        configData->convergence = false;
    }

    // Store the current simulation time
    double t = callTime * NANO2SEC;

    // Define the scalar prescribed states
    double thetaDDot;
    double thetaDot;
    double theta;

    // Compute the prescribed scalar states at the current simulation time
    if ((t < configData->ts || t == configData->ts) && configData->tf - configData->tInit != 0) // Entered during the first half of the maneuver
    {
        thetaDDot = configData->thetaDDotMax;
        thetaDot = thetaDDot * (t - configData->tInit) + configData->thetaDotInit;
        theta = configData->a * (t - configData->tInit) * (t - configData->tInit) + configData->thetaInit;
    }
    else if ( t > configData->ts && t <= configData->tf && configData->tf - configData->tInit != 0) // Entered during the second half of the maneuver
    {
        thetaDDot = -1 * configData->thetaDDotMax;
        thetaDot = thetaDDot * (t - configData->tInit) + configData->thetaDotInit - thetaDDot * (configData->tf - configData->tInit);
        theta = configData->b * (t - configData->tf) * (t - configData->tf) + configData->thetaRef;
    }
    else // Entered when the maneuver is complete
    {
        thetaDDot = 0.0;
        thetaDot = configData->thetaDotRef;
        theta = configData->thetaRef;
        configData->convergence = true;
    }

    // Determine dcm_FF0
    double dcm_FF0[3][3];
    double prv_FF0_array[3];
    double theta_FF0 = theta - configData->thetaInit;
    v3Scale(theta_FF0, configData->rotAxis_M, prv_FF0_array);
    PRV2C(prv_FF0_array, dcm_FF0);

    // Determine dcm_F0M
    double dcm_F0M[3][3];
    double prv_F0M_array[3];
    v3Scale(configData->thetaInit, configData->rotAxis_M, prv_F0M_array);
    PRV2C(prv_F0M_array, dcm_F0M);

    // Determine dcm_FM
    double dcm_FM[3][3];
    m33MultM33(dcm_FF0, dcm_F0M, dcm_FM);

    // Determine the prescribed parameter: sigma_FM
    C2MRP(dcm_FM, configData->sigma_FM);

    // Copy the module variables to the prescribedRotationOut output message
    v3Copy(configData->omega_FM_F, prescribedRotationOut.omega_FM_F);
    v3Copy(configData->omegaPrime_FM_F, prescribedRotationOut.omegaPrime_FM_F);
    v3Copy(configData->sigma_FM, prescribedRotationOut.sigma_FM);

    // Copy the local scalar variables to the spinningBodyOut output message
    spinningBodyOut.theta = theta;
    spinningBodyOut.thetaDot = thetaDot;

    // Write the output messages
    HingedRigidBodyMsg_C_write(&spinningBodyOut, &configData->spinningBodyOutMsg, moduleID, callTime);
    PrescribedRotationMsg_C_write(&prescribedRotationOut, &configData->prescribedRotationOutMsg, moduleID, callTime);
}
