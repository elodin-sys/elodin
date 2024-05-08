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

#include "fswAlgorithms/effectorInterfaces/rwNullSpace/rwNullSpace.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*!
 \verbatim embed:rst
    This method creates the module output message of type :ref:`ArrayMotorTorqueMsgPayload`.
 \endverbatim
 @return void
 @param configData The configuration data associated with RW null space model
 @param moduleID The ID associated with the configData
 */
void SelfInit_rwNullSpace(rwNullSpaceConfig *configData, int64_t moduleID)
{
    ArrayMotorTorqueMsg_C_init(&configData->rwMotorTorqueOutMsg);
}


/*! @brief This resets the module to original states by reading in the RW configuration messages and recreating any module specific variables.  The output message is reset to zero.
    @return void
    @param configData The configuration data associated with the null space control
    @param callTime The clock time at which the function was called (nanoseconds)
    @param moduleID The ID associated with the configData
 */
void Reset_rwNullSpace(rwNullSpaceConfig *configData, uint64_t callTime,
                        int64_t moduleID)
{
    double GsMatrix[3*MAX_EFF_CNT];                 /* [-]  [Gs] projection matrix where gs_hat_B RW spin axis form each colum */
    double GsTranspose[3 * MAX_EFF_CNT];            /* [-]  [Gs]^T */
    double GsInvHalf[3 * 3];                        /* [-]  ([Gs][Gs]^T)^-1 */
    double identMatrix[MAX_EFF_CNT*MAX_EFF_CNT];    /* [-]  [I_NxN] identity matrix */
    double GsTemp[MAX_EFF_CNT*MAX_EFF_CNT];         /* [-]  temp matrix */
    RWConstellationMsgPayload localRWData;          /*      local copy of RW configuration data */

    // check if the required input messages are included
    if (!RWConstellationMsg_C_isLinked(&configData->rwConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwNullSpace.rwConfigInMsg wasn't connected.");
    }
    if (!ArrayMotorTorqueMsg_C_isLinked(&configData->rwMotorTorqueInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwNullSpace.rwMotorTorqueInMsg wasn't connected.");
    }
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwNullSpace.rwSpeedsInMsg wasn't connected.");
    }

    /* read in the RW spin axis headings */
    localRWData = RWConstellationMsg_C_read(&configData->rwConfigInMsg);

    /* create the 3xN [Gs] RW spin axis projection matrix */
    configData->numWheels = (uint32_t) localRWData.numRW;
    if (configData->numWheels > MAX_EFF_CNT) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwNullSpace.numWheels is larger that max effector count.");
    }
    for(uint32_t i=0; i<configData->numWheels; i=i+1)
    {
        for(int j=0; j<3; j=j+1)
        {
            GsMatrix[j*(int) configData->numWheels+i] = localRWData.reactionWheels[i].gsHat_B[j];
        }
    }

    /* find the [tau] null space projection matrix [tau]= ([I] - [Gs]^T.([Gs].[Gs]^T) */
    mTranspose(GsMatrix, 3, configData->numWheels, GsTranspose);            /* find [Gs]^T */
    mMultM(GsMatrix, 3, configData->numWheels, GsTranspose,
           configData->numWheels, 3, GsInvHalf);                            /* find [Gs].[Gs]^T */
    m33Inverse(RECAST3X3 GsInvHalf, RECAST3X3 GsInvHalf);                   /* find ([Gs].[Gs]^T)^-1 */
    mMultM(GsInvHalf, 3, 3, GsMatrix, 3, configData->numWheels,
           configData->tau);                                                /* find ([Gs].[Gs]^T)^-1.[Gs] */
    mMultM(GsTranspose, configData->numWheels, 3, configData->tau, 3,
           configData->numWheels, GsTemp);                                  /* find [Gs]^T.([Gs].[Gs]^T)^-1.[Gs] */
    mSetIdentity(identMatrix, configData->numWheels, configData->numWheels);
    mSubtract(identMatrix, configData->numWheels, configData->numWheels,    /* find ([I] - [Gs]^T.([Gs].[Gs]^T)^-1.[Gs]) */
              GsTemp, configData->tau);

}

/*! This method takes the input reaction wheel commands as well as the observed 
    reaction wheel speeds and balances the commands so that the overall vehicle 
	momentum is minimized.
 @return void
 @param configData The configuration data associated with the null space control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_rwNullSpace(rwNullSpaceConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    ArrayMotorTorqueMsgPayload cntrRequest;        /* [Nm]  array of the RW motor torque solution vector from the control module */
    RWSpeedMsgPayload rwSpeeds;                    /* [r/s] array of RW speeds */
    RWSpeedMsgPayload rwDesiredSpeeds;             /* [r/s] array of RW speeds */
	ArrayMotorTorqueMsgPayload finalControl;       /* [Nm]  array of final RW motor torques containing both
                                                       the control and null motion torques */
	double dVector[MAX_EFF_CNT];                   /* [Nm]  null motion wheel speed control array */
    double DeltaOmega[MAX_EFF_CNT];                /* [r/s] difference in RW speeds */
    
    /* zero all output message containers prior to evaluation */
    finalControl = ArrayMotorTorqueMsg_C_zeroMsgPayload();

    /* Read the input RW commands to get the raw RW requests*/
    cntrRequest = ArrayMotorTorqueMsg_C_read(&configData->rwMotorTorqueInMsg);
    /* Read the RW speeds*/
    rwSpeeds = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);

    /* make the default desired wheel speed zero and read in values if connected */
    rwDesiredSpeeds = RWSpeedMsg_C_zeroMsgPayload();
    if (RWSpeedMsg_C_isLinked(&configData->rwDesiredSpeedsInMsg)) {
        rwDesiredSpeeds = RWSpeedMsg_C_read(&configData->rwDesiredSpeedsInMsg);
    }

    /* compute the wheel speed control vector d = -K.DeltaOmega */
    vSubtract(rwSpeeds.wheelSpeeds, configData->numWheels, rwDesiredSpeeds.wheelSpeeds, DeltaOmega);
	vScale(-configData->OmegaGain, DeltaOmega, configData->numWheels, dVector);

    /* compute the RW null space motor torque solution to reduce the wheel speeds */
	mMultV(configData->tau, configData->numWheels, configData->numWheels,
		dVector, finalControl.motorTorque);
    
    /* add the null motion RW torque solution to the RW feedback control torque solution */
	vAdd(finalControl.motorTorque, configData->numWheels,
		cntrRequest.motorTorque, finalControl.motorTorque);

    /* write the final RW torque solution to the output message */
    ArrayMotorTorqueMsg_C_write(&finalControl, &configData->rwMotorTorqueOutMsg, moduleID, callTime);

    return;
}
