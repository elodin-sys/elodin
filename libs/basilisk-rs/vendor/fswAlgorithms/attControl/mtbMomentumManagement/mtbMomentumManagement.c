/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "mtbMomentumManagement.h"
#include "string.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/svd.h"
#include <stdio.h>

/*!
 \verbatim embed:rst
    This method initializes the configData for this module.
    It checks to ensure that the inputs are sane and then creates the
    output message of type :ref:`MTBCmdMsgPayload` and :ref:`ArrayMotorTorqueMsgPayload`.
 \endverbatim
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_mtbMomentumManagement(mtbMomentumManagementConfig *configData, int64_t moduleID)
{
    MTBCmdMsg_C_init(&configData->mtbCmdOutMsg);
    ArrayMotorTorqueMsg_C_init(&configData->rwMotorTorqueOutMsg);

    return;
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.  The local copy of the
 message output buffer should be cleared.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_mtbMomentumManagement(mtbMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Check if the required input messages are linked.
     */
    if (!RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.rwParamsInMsg is not connected.");
    }
    if(!MTBArrayConfigMsg_C_isLinked(&configData->mtbParamsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.mtbParamsInMsg is not connected.");
    }
    if (!TAMSensorBodyMsg_C_isLinked(&configData->tamSensorBodyInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.tamSensorBodyInMsg is not connected.");
    }
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.rwSpeedsInMsg is not connected.");
    }
    if (!ArrayMotorTorqueMsg_C_isLinked(&configData->rwMotorTorqueInMsg)){
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mtbMomentumManagement.rwMotorTorqueInMsg is not connected.");
    }
    
    /*! - Read the input configuration messages.*/
    configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);
    configData->mtbConfigParams = MTBArrayConfigMsg_C_read(&configData->mtbParamsInMsg);

    return;
}


/*! Computes the appropriate wheel torques and magnetic torque bar dipoles to bias the wheels to their desired speeds.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_mtbMomentumManagement(mtbMomentumManagementConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*
     * Declare and initialize local variables.
     */
    int numRW = configData->rwConfigParams.numRW;
    int numMTB = configData->mtbConfigParams.numMTB;
    int j = 0;
    double BTilde_B[3*3];
    double BGt[3*MAX_EFF_CNT];
    double BGtPsuedoInverse[MAX_EFF_CNT*3];
    double uDelta_B[3];
    double uDelta_W[MAX_EFF_CNT];
    double GsPsuedoInverse[MAX_EFF_CNT*3];
    double Gs[3 * MAX_EFF_CNT];
    mSetZero(BTilde_B, 3, 3);
    mSetZero(BGt, 3, numMTB);
    mSetZero(BGtPsuedoInverse, numMTB, 3);
    v3SetZero(uDelta_B);
    vSetZero(uDelta_W, numRW);
    mSetZero(GsPsuedoInverse, numRW, 3);
    mTranspose(configData->rwConfigParams.GsMatrix_B, configData->rwConfigParams.numRW, 3, Gs);
    v3SetZero(configData->tauDesiredMTB_B);
    v3SetZero(configData->tauDesiredRW_B);
    vSetZero(configData->hDeltaWheels_W, numRW);
    v3SetZero(configData->hDeltaWheels_B);
    vSetZero(configData->tauDesiredRW_W, numRW);
    vSetZero(configData->tauIdealRW_W, numRW);
    v3SetZero(configData->tauIdealRW_B);
    vSetZero(configData->wheelSpeedError_W, numRW);

    /*
     * Read input messages and initialize output messages.
     */
    TAMSensorBodyMsgPayload tamSensorBodyInMsgBuffer = TAMSensorBodyMsg_C_read(&configData->tamSensorBodyInMsg);
    RWSpeedMsgPayload rwSpeedsInMsgBuffer = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);
    ArrayMotorTorqueMsgPayload rwMotorTorqueInMsgBuffer = ArrayMotorTorqueMsg_C_read(&configData->rwMotorTorqueInMsg);
    MTBCmdMsgPayload mtbCmdOutputMsgBuffer = MTBCmdMsg_C_zeroMsgPayload();
    ArrayMotorTorqueMsgPayload rwMotorTorqueOutMsgBuffer = rwMotorTorqueInMsgBuffer;
    
    /*! - Compute the wheel speed feedback.*/
    vSubtract(rwSpeedsInMsgBuffer.wheelSpeeds, numRW, configData->wheelSpeedBiases, configData->wheelSpeedError_W);
    vElementwiseMult(configData->rwConfigParams.JsList, numRW, configData->wheelSpeedError_W, configData->hDeltaWheels_W);
    mMultV(Gs, 3, numRW, configData->hDeltaWheels_W, configData->hDeltaWheels_B);
    vScale(-configData->cGain, configData->hDeltaWheels_W, numRW, configData->tauIdealRW_W);
    mMultV(Gs, 3, numRW, configData->tauIdealRW_W, configData->tauIdealRW_B);

    /*! - Compute the magnetic torque bar dipole commands. */
    v3TildeM(tamSensorBodyInMsgBuffer.tam_B, BTilde_B);
    mMultM(BTilde_B, 3, 3, configData->mtbConfigParams.GtMatrix_B, 3, numMTB, BGt);
    solveSVD(BGt, 3, numMTB, mtbCmdOutputMsgBuffer.mtbDipoleCmds, configData->tauIdealRW_B, 0.00000000001);
    vScale(-1.0, mtbCmdOutputMsgBuffer.mtbDipoleCmds, numMTB, mtbCmdOutputMsgBuffer.mtbDipoleCmds);
    
    /*
     * Saturate dipoles.
     */
    for (j = 0; j < numMTB; j++)
    {
        if (mtbCmdOutputMsgBuffer.mtbDipoleCmds[j] > configData->mtbConfigParams.maxMtbDipoles[j])
            mtbCmdOutputMsgBuffer.mtbDipoleCmds[j] = configData->mtbConfigParams.maxMtbDipoles[j];
        
        if (mtbCmdOutputMsgBuffer.mtbDipoleCmds[j] < -configData->mtbConfigParams.maxMtbDipoles[j])
            mtbCmdOutputMsgBuffer.mtbDipoleCmds[j] = -configData->mtbConfigParams.maxMtbDipoles[j];
    }
    
    /*! - Compute the desired Body torque produced by the torque bars.*/
    mMultV(BGt, 3, numMTB, mtbCmdOutputMsgBuffer.mtbDipoleCmds, configData->tauDesiredMTB_B);
    vScale(-1.0, configData->tauDesiredMTB_B, 3, configData->tauDesiredMTB_B);
    
    /*! - Compute the reaction wheel torque commands.*/
    v3Subtract(configData->tauDesiredMTB_B, configData->tauIdealRW_B, uDelta_B);
    mMinimumNormInverse(Gs, 3, numRW, GsPsuedoInverse);
    mMultV(GsPsuedoInverse, numRW, 3, uDelta_B, uDelta_W);
    vAdd(configData->tauIdealRW_W, numRW, uDelta_W, configData->tauDesiredRW_W);
    
    /*! - Compute the desired Body torque produced by the reaction wheels.*/
    mMultV(Gs, 3, numRW, configData->tauDesiredRW_W, configData->tauDesiredRW_B);
    vScale(-1.0, configData->tauDesiredRW_B, 3, configData->tauDesiredRW_B);
    
    /*
     * Write output messages.
     */
    MTBCmdMsg_C_write(&mtbCmdOutputMsgBuffer, &configData->mtbCmdOutMsg, moduleID, callTime);
    vAdd(configData->tauDesiredRW_W, numRW, rwMotorTorqueOutMsgBuffer.motorTorque, rwMotorTorqueOutMsgBuffer.motorTorque);
    ArrayMotorTorqueMsg_C_write(&rwMotorTorqueOutMsgBuffer, &configData->rwMotorTorqueOutMsg, moduleID, callTime);
    
    return;
}

/*
 * Returns the tilde matrix in the form of a 1-D array.
 */
void v3TildeM(double v[3], void *result)
{
    double *m_result = (double *)result;
    m_result[MXINDEX(3, 0, 0)] = 0.0;
    m_result[MXINDEX(3, 0, 1)] = -v[2];
    m_result[MXINDEX(3, 0, 2)] = v[1];
    m_result[MXINDEX(3, 1, 0)] = v[2];
    m_result[MXINDEX(3, 1, 1)] = 0.0;
    m_result[MXINDEX(3, 1, 2)] = -v[0];
    m_result[MXINDEX(3, 2, 0)] = -v[1];
    m_result[MXINDEX(3, 2, 1)] = v[0];
    m_result[MXINDEX(3, 2, 2)] = 0.0;
}
