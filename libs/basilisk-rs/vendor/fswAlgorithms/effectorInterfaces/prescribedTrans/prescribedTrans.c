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

/*! Import the module header file */
#include "prescribedTrans.h"

/*! Import other required files */
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/macroDefinitions.h"

/*! This method initializes the output message for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_prescribedTrans(PrescribedTransConfig *configData, int64_t moduleID)
{
    // Initialize the module output message
    PrescribedTranslationMsg_C_init(&configData->prescribedTranslationOutMsg);
}

/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values. This method also checks
 if the module input message is linked.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Reset_prescribedTrans(PrescribedTransConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Check if the input message is connected
    if (!LinearTranslationRigidBodyMsg_C_isLinked(&configData->linearTranslationRigidBodyInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: prescribedTrans.linearTranslationRigidBodyInMsg wasn't connected.");
    }

    // Set the initial time
    configData->tInit = 0.0;

    // Set the initial convergence to true to enter the correct loop in the Update() method on the first pass
    configData->convergence = true;
}

/*! This method uses the given initial and reference attitudes to compute the required attitude maneuver as
a function of time. The profiled translational trajectory is updated in time and written to the module's prescribed
motion output message.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] Time the method is called
 @param moduleID The module identifier
*/
void Update_prescribedTrans(PrescribedTransConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // Create the buffer messages
    LinearTranslationRigidBodyMsgPayload linearTranslationRigidBodyIn;
    PrescribedTranslationMsgPayload prescribedTranslationOut;

    // Zero the output message
    prescribedTranslationOut = PrescribedTranslationMsg_C_zeroMsgPayload();

    // Read the input message
    linearTranslationRigidBodyIn = LinearTranslationRigidBodyMsg_C_zeroMsgPayload();
    if (LinearTranslationRigidBodyMsg_C_isWritten(&configData->linearTranslationRigidBodyInMsg))
    {
        linearTranslationRigidBodyIn = LinearTranslationRigidBodyMsg_C_read(&configData->linearTranslationRigidBodyInMsg);
    }

    // This loop is entered when a new maneuver is requested after all previous maneuvers are completed
    if (LinearTranslationRigidBodyMsg_C_timeWritten(&configData->linearTranslationRigidBodyInMsg) <= callTime && configData->convergence)
    {
        // Store the initial information
        configData->tInit = callTime * NANO2SEC;
        configData->scalarPosInit = v3Norm(configData->r_FM_M);
        configData->scalarVelInit = v3Norm(configData->rPrime_FM_M);

        // Store the reference information
        configData->scalarPosRef = linearTranslationRigidBodyIn.rho;
        configData->scalarVelRef = 0.0;

        // Define temporal information
        double convTime = sqrt(((0.5 * fabs(configData->scalarPosRef - configData->scalarPosInit)) * 8) / configData->scalarAccelMax);
        configData->tf = configData->tInit + convTime;
        configData->ts = configData->tInit + convTime / 2;

        // Define the parabolic constants for the maneuver
        configData->a = 0.5 * (configData->scalarPosRef - configData->scalarPosInit) / ((configData->ts - configData->tInit) * (configData->ts - configData->tInit));
        configData->b = -0.5 * (configData->scalarPosRef - configData->scalarPosInit) / ((configData->ts - configData->tf) * (configData->ts - configData->tf));

        // Set the convergence to false to execute the maneuver
        configData->convergence = false;
    }

    // Store the current simulation time
    double t = callTime * NANO2SEC;

    // Define scalar prescribed states
    double scalarAccel;
    double scalarVel;
    double scalarPos;

    // Compute the prescribed scalar states: scalarAccel, scalarVel, and scalarPos
    if ((t < configData->ts || t == configData->ts) && configData->tf - configData->tInit != 0)  // Entered during the first half of the maneuver
    {
        scalarAccel = configData->scalarAccelMax;
        scalarVel = scalarAccel * (t - configData->tInit) + configData->scalarVelInit;
        scalarPos = configData->a * (t - configData->tInit) * (t - configData->tInit) + configData->scalarPosInit;
    }
    else if ( t > configData->ts && t <= configData->tf && configData->tf - configData->tInit != 0)  // Entered during the second half of the maneuver
    {
        scalarAccel = -1 * configData->scalarAccelMax;
        scalarVel = scalarAccel * (t - configData->tInit) + configData->scalarVelInit - scalarAccel * (configData->tf - configData->tInit);
        scalarPos = configData->b * (t - configData->tf) * (t - configData->tf) + configData->scalarPosRef;
    }
    else  // Entered when the maneuver is complete
    {
        scalarAccel = 0.0;
        scalarVel = configData->scalarVelRef;
        scalarPos = configData->scalarPosRef;
        configData->convergence = true;
    }

    // Convert the scalar variables to the prescribed parameters
    v3Scale(scalarPos, configData->transAxis_M, configData->r_FM_M);
    v3Scale(scalarVel, configData->transAxis_M, configData->rPrime_FM_M);
    v3Scale(scalarAccel, configData->transAxis_M, configData->rPrimePrime_FM_M);

    // Copy the local variables to the output message
    v3Copy(configData->r_FM_M, prescribedTranslationOut.r_FM_M);
    v3Copy(configData->rPrime_FM_M, prescribedTranslationOut.rPrime_FM_M);
    v3Copy(configData->rPrimePrime_FM_M, prescribedTranslationOut.rPrimePrime_FM_M);

    // Write the prescribed motion output message
    PrescribedTranslationMsg_C_write(&prescribedTranslationOut, &configData->prescribedTranslationOutMsg, moduleID, callTime);
}
