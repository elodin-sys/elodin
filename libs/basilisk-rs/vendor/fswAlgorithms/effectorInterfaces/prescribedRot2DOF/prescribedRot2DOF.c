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

/* Include the module header file. */
#include "prescribedRot2DOF.h"

/* Import other required files. */
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/rigidBodyKinematics.h"

/*! This method initializes the output message for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, int64_t moduleID)
{
    PrescribedRotationMsg_C_init(&configData->prescribedRotationOutMsg);
}

/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values. A check is also
 performed to ensure the user sets the configurable module variables.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Reset_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Check if the required input messages are linked */
    if (!HingedRigidBodyMsg_C_isLinked(&configData->spinningBodyRef1InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "prescribedRot2DOF.spinningBodyRef1InMsg wasn't connected.");
    }

    if (!HingedRigidBodyMsg_C_isLinked(&configData->spinningBodyRef2InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "prescribedRot2DOF.spinningBodyRef2InMsg wasn't connected.");
    }

    // Check that the user-configurable variables are set
    if (configData->phiDDotMax < 0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "prescribedRot2DOF.phiDDotMax wasn't set.");
    }

    if (v3Norm(configData->rotAxis1_M) < 1e-6) {
        _bskLog(configData->bskLogger, BSK_ERROR, "prescribedRot2DOF.rotAxis1_M wasn't set.");
    }

    if (v3Norm(configData->rotAxis2_F1) < 1e-6) {
        _bskLog(configData->bskLogger, BSK_ERROR, "prescribedRot2DOF.rotAxis2_F1 wasn't set.");
    }

    // Store the initial time */
    configData->maneuverStartTime = callTime * NANO2SEC;

    // Set the initial convergence to true to properly enter the desired loop in the Update() method on the first pass
    configData->isManeuverComplete = true;

    // Zero the PRV angle variables
    configData->phiRef = 0.0;
    configData->phiRefAccum = 0.0;
    configData->phiAccum = 0.0;
}

/*! This method profiles a 1DOF rotational trajectory given two rotation angles and rotation axes. The prescribed states
are updated in this routine as a function of time and written to the prescribedMotion output message.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Update_prescribedRot2DOF(PrescribedRot2DOFConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Create buffer messages
    HingedRigidBodyMsgPayload spinningBodyRef1In;
    HingedRigidBodyMsgPayload spinningBodyRef2In;
    HingedRigidBodyMsgPayload spinningBodyOut;
    PrescribedRotationMsgPayload prescribedRotationOut;

    // Zero the output messages
    spinningBodyOut = HingedRigidBodyMsg_C_zeroMsgPayload();
    prescribedRotationOut = PrescribedRotationMsg_C_zeroMsgPayload();

    // Read the input messages
    spinningBodyRef1In = HingedRigidBodyMsg_C_zeroMsgPayload();
    if (HingedRigidBodyMsg_C_isWritten(&configData->spinningBodyRef1InMsg))
    {
        spinningBodyRef1In = HingedRigidBodyMsg_C_read(&configData->spinningBodyRef1InMsg);
    }

    spinningBodyRef2In = HingedRigidBodyMsg_C_zeroMsgPayload();
    if (HingedRigidBodyMsg_C_isWritten(&configData->spinningBodyRef2InMsg))
    {
        spinningBodyRef2In = HingedRigidBodyMsg_C_read(&configData->spinningBodyRef2InMsg);
    }

    /* This loop is entered when the spinning body attitude converges to the reference attitude. The PRV angle and axis
     reference parameters are updated along with the profiled trajectory parameters. */
    if ((HingedRigidBodyMsg_C_timeWritten(&configData->spinningBodyRef1InMsg) <= callTime
    || HingedRigidBodyMsg_C_timeWritten(&configData->spinningBodyRef2InMsg) <= callTime ) && configData->isManeuverComplete)
    {
        // Define the initial time
        configData->maneuverStartTime = callTime * NANO2SEC;

        // Calculate dcm_F0M. This DCM represents the current spinning body attitude with respect to the M frame
        double dcm_FM[3][3];
        MRP2C(configData->sigma_FM, dcm_FM);
        m33Copy(dcm_FM, configData->dcm_F0M);

        // Store the reference variables from the spinningBody input messages
        double theta1Ref = spinningBodyRef1In.theta;
        double theta2Ref = spinningBodyRef2In.theta;

        // Convert the reference angles and their associated rotation axes to PRVs
        double prv_F1M_array[3];                    // 1st PRV representing the intermediate frame relative to the M frame
        double prv_F2F1_array[3];                   // 2nd PRV representing the final reference frame relative to the intermediate frame
        v3Normalize(configData->rotAxis1_M, configData->rotAxis1_M);
        v3Normalize(configData->rotAxis2_F1, configData->rotAxis2_F1);
        v3Scale(theta1Ref, configData->rotAxis1_M, prv_F1M_array);
        v3Scale(theta2Ref, configData->rotAxis2_F1, prv_F2F1_array);

        // Convert the reference PRVs to DCMs
        double dcm_F1M[3][3];                       // 1st DCM representing the intermediate frame relative to the M frame
        double dcm_F2F1[3][3];                      // 2nd DCM representing the final reference frame relative to the intermediate frame
        PRV2C(prv_F1M_array, dcm_F1M);
        PRV2C(prv_F2F1_array, dcm_F2F1);

        // Combine the two reference DCMs to a single reference DCM
        double dcm_F2M[3][3];                       // DCM representing the final reference frame relative to the M frame
        m33MultM33(dcm_F2F1, dcm_F1M, dcm_F2M);

        // Convert dcm_F2M to a PRV
        double prv_F2M_array[3];                    // PRV representing the final reference frame relative to the M frame
        C2PRV(dcm_F2M, prv_F2M_array);

        // Determine dcm_F2F. This DCM represents the final reference attitude with respect to the current spinning body body frame
        double dcm_F2F[3][3];
        m33MultM33t(dcm_F2M, dcm_FM, dcm_F2F);

        // Convert dcm_F2F to a PRV
        double prv_F2F_array[3];                    // PRV representing the final reference frame relative to the current spinning body body frame
        C2PRV(dcm_F2F, prv_F2F_array);

        // Compute the single PRV reference angle for the attitude maneuver.
        v3Normalize(prv_F2F_array, configData->rotAxis_M);
        configData->phiRef = v3Dot(prv_F2F_array, configData->rotAxis_M);

        // Store the accumulated reference PRVs
        configData->phiRefAccum = configData->phiAccum;

        // Define temporal information
        double convTime = sqrt(fabs(configData->phiRef) * 4 / configData->phiDDotMax); // Time for the individual attitude maneuver
        configData->maneuverEndTime = configData->maneuverStartTime + convTime;
        configData->maneuverSwitchTime = convTime / 2 + configData->maneuverStartTime;

        // Define the maneuver parabolic constants
        configData->a = 0.5 * configData->phiRef / ((configData->maneuverSwitchTime - configData->maneuverStartTime) * (configData->maneuverSwitchTime - configData->maneuverStartTime));
        configData->b = -0.5 * configData->phiRef / ((configData->maneuverSwitchTime - configData->maneuverEndTime) * (configData->maneuverSwitchTime - configData->maneuverEndTime));

        // Set the convergence to false until the attitude maneuver is complete
        configData->isManeuverComplete = false;
    }

    // Store the current simulation time
    double t = callTime * NANO2SEC; // [s]

    // Define the other scalar module states locally
    double phiDDot;
    double phiDot;

    // Compute the prescribed states at the current time for the profiled trajectory
    if ((t < configData->maneuverSwitchTime || t == configData->maneuverSwitchTime) && configData->maneuverEndTime != configData->maneuverStartTime) // Entered during the first half of the attitude maneuver
    {
        phiDDot = configData->phiDDotMax;
        phiDot = phiDDot * (t - configData->maneuverStartTime);
        configData->phi = configData->a * (t - configData->maneuverStartTime) * (t - configData->maneuverStartTime);
    }
    else if ( t > configData->maneuverSwitchTime && t <= configData->maneuverEndTime && configData->maneuverEndTime != configData->maneuverStartTime) // Entered during the second half of the attitude maneuver
    {
        phiDDot = -1 * configData->phiDDotMax;
        phiDot = phiDDot * (t - configData->maneuverEndTime );
        configData->phi = configData->b * (t - configData->maneuverEndTime) * (t - configData->maneuverEndTime) + configData->phiRef;
    }
    else // Entered if the maneuver is complete
    {
        phiDDot = 0.0;
        phiDot = 0.0;
        configData->phi = configData->phiRef;
        configData->isManeuverComplete = true;
    }

    // Store the accumulated PRV angle
    configData->phiAccum = configData->phiRefAccum + configData->phi;

    // Determine the prescribed spinning body states: omega_FM_F and omegaPrime_FM_F
    v3Normalize(configData->rotAxis_M, configData->rotAxis_M);
    v3Scale(phiDot, configData->rotAxis_M, configData->omega_FM_F);
    v3Scale(phiDDot, configData->rotAxis_M, configData->omegaPrime_FM_F);

    // Calculate PRV representing the current spinning body attitude with respect to its initial attitude
    double prv_FF0_array[3];
    v3Scale(configData->phi, configData->rotAxis_M, prv_FF0_array);

    // Determine dcm_FF0. This DCM represents the current spinning body attitude with respect to its initial attitude
    double dcm_FF0[3][3];
    PRV2C(prv_FF0_array, dcm_FF0);

    // Determine dcm_FM. This DCM represents the current spinning body attitude with respect to the M frame
    double dcm_FM[3][3];
    m33MultM33(dcm_FF0, configData->dcm_F0M, dcm_FM);

    // Determine the prescribed spinning body state: sigma_FM
    C2MRP(dcm_FM, configData->sigma_FM);

    // Copy the module prescribed variables to the prescribed rotational motion output message
    v3Copy(configData->omega_FM_F, prescribedRotationOut.omega_FM_F);
    v3Copy(configData->omegaPrime_FM_F, prescribedRotationOut.omegaPrime_FM_F);
    v3Copy(configData->sigma_FM, prescribedRotationOut.sigma_FM);

    // Write the prescribed rotational motion output message
    PrescribedRotationMsg_C_write(&prescribedRotationOut, &configData->prescribedRotationOutMsg, moduleID, callTime);
}
