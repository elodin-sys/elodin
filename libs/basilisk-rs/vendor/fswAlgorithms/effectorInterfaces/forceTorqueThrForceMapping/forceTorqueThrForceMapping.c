/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "fswAlgorithms/effectorInterfaces/forceTorqueThrForceMapping/forceTorqueThrForceMapping.h"
#include "string.h"
#include "architecture/utilities/linearAlgebra.h"

/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig  *configData, int64_t moduleID)
{
    THRArrayCmdForceMsg_C_init(&configData->thrForceCmdOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
    time varying states between function calls are reset to their default values.
    Check if required input messages are connected.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!THRArrayConfigMsg_C_isLinked(&configData->thrConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: forceTorqueThrForceMapping.thrConfigInMsg was not connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->vehConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: forceTorqueThrForceMapping.vehConfigInMsg was not connected.");
    }

    VehicleConfigMsgPayload vehConfigInMsgBuffer;  //!< local copy of message buffer
    THRArrayConfigMsgPayload thrConfigInMsgBuffer;  //!< local copy of message buffer

    //!< read the rest of the input messages
    thrConfigInMsgBuffer = THRArrayConfigMsg_C_read(&configData->thrConfigInMsg);
    vehConfigInMsgBuffer = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);

    /*! - copy the thruster position and thruster force heading information into the module configuration data */
    configData->numThrusters = (uint32_t) thrConfigInMsgBuffer.numThrusters;
    v3Copy(vehConfigInMsgBuffer.CoM_B, configData->CoM_B);
    if (configData->numThrusters > MAX_EFF_CNT) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: forceTorqueThrForceMapping thruster configuration input message has a number of thrusters that is larger than MAX_EFF_CNT");
    }

    /*! - copy the thruster position and thruster force heading information into the module configuration data */
    for(uint32_t i = 0; i < configData->numThrusters; i++)
    {
        v3Copy(thrConfigInMsgBuffer.thrusters[i].rThrust_B, configData->rThruster_B[i]);
        v3Copy(thrConfigInMsgBuffer.thrusters[i].tHatThrust_B, configData->gtThruster_B[i]);
        if(thrConfigInMsgBuffer.thrusters[i].maxThrust <= 0.0){
            _bskLog(configData->bskLogger, BSK_ERROR, "Error: forceTorqueThrForceMapping: A configured thruster has a non-sensible saturation limit of <= 0 N!");
        }
    }
}


/*! Add a description of what this main Update() routine does for this module
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    CmdTorqueBodyMsgPayload cmdTorqueInMsgBuffer;  //!< local copy of message buffer
    CmdForceBodyMsgPayload cmdForceInMsgBuffer;  //!< local copy of message buffer
    THRArrayCmdForceMsgPayload thrForceCmdOutMsgBuffer;  //!< local copy of message buffer

    // always zero the output message buffers before assigning values
    thrForceCmdOutMsgBuffer = THRArrayCmdForceMsg_C_zeroMsgPayload();

    /* Check if torque message is linked and read, zero out if not*/
    if (CmdTorqueBodyMsg_C_isLinked(&configData->cmdTorqueInMsg)) {
        cmdTorqueInMsgBuffer = CmdTorqueBodyMsg_C_read(&configData->cmdTorqueInMsg);
    } else{
        cmdTorqueInMsgBuffer = CmdTorqueBodyMsg_C_zeroMsgPayload();
    }

    /* Check if force message is linked and read, zero out if not*/
    if (CmdForceBodyMsg_C_isLinked(&configData->cmdForceInMsg)) {
        cmdForceInMsgBuffer = CmdForceBodyMsg_C_read(&configData->cmdForceInMsg);
    } else{
        cmdForceInMsgBuffer = CmdForceBodyMsg_C_zeroMsgPayload();
    }

    /* Initialize variables */
    double DG[6][MAX_EFF_CNT];
    double rThrusterRelCOM_B[MAX_EFF_CNT][3];
    double rCrossGt[3];
    double zeroVector[MAX_EFF_CNT];
    uint32_t zeroRows[6];
    uint32_t numZeroes;
    double force_B[MAX_EFF_CNT];
    double forceTorque_B[6];
    double forceSubtracted_B[MAX_EFF_CNT];
    vSetZero(force_B, (size_t) MAX_EFF_CNT);
    vSetZero(forceSubtracted_B, (size_t) MAX_EFF_CNT);

    for (uint32_t i = 0; i < 6; i++) {
        for (uint32_t j = 0; j < MAX_EFF_CNT; j++) {
            DG[i][j] = 0.0;
        }
    }

    /* Create the torque and force vector */
    for (uint32_t i = 0; i < 3; i++) {
        forceTorque_B[i] = cmdTorqueInMsgBuffer.torqueRequestBody[i];
        forceTorque_B[i+3] = cmdForceInMsgBuffer.forceRequestBody[i];
    }

    /* - compute thruster locations relative to COM */
    for (uint32_t i = 0; i<configData->numThrusters; i++) {
        v3Subtract(configData->rThruster_B[i], configData->CoM_B, rThrusterRelCOM_B[i]);
    }

    /* Fill DG with thruster directions and moment arms */
    for (uint32_t i = 0; i < configData->numThrusters; i++) {
        /* Compute moment arm and fill in */
        v3Cross(rThrusterRelCOM_B[i], configData->gtThruster_B[i], rCrossGt);
        for(uint32_t j = 0; j < 3; j++) {
            DG[j][i] = rCrossGt[j];
        }

        /* Fill in control axes */
        for(uint32_t j = 0; j < 3; j++) {
            DG[j+3][i] = configData->gtThruster_B[i][j];
        }
    }

    /* Check DG for zero rows */
    vSetZero(zeroVector, configData->numThrusters);
    numZeroes = 0;
    for(uint32_t j = 0; j < 6; j++) {
        if (vIsEqual(zeroVector, 6, DG[j], 0.0000001)) {
            zeroRows[j] = 1;
            numZeroes += 1;
        } else {
            zeroRows[j] = 0;
        }
    }

    /* Create the DG w/ zero rows removed */
    double DG_full[6*MAX_EFF_CNT];
    vSetZero(DG_full, (size_t) 6*MAX_EFF_CNT);
    uint32_t zeroesPassed;
    zeroesPassed = 0;
    for(uint32_t i = 0; i < 6; i++) {
        if (!zeroRows[i]) {
            for(uint32_t j = 0; j < MAX_EFF_CNT; j++) {
                DG_full[MXINDEX(MAX_EFF_CNT, i-zeroesPassed, j)] = DG[i][j];
            }
        } else {
            zeroesPassed += 1;
        }
    }

    /* Compute the minimum norm inverse of DG*/
    double DGT_DGDGT_inv[6*6];
    mMinimumNormInverse(DG_full, (size_t) 6-numZeroes, (size_t) MAX_EFF_CNT, DGT_DGDGT_inv);

    /* Compute the force for each thruster */
    mMultV(DGT_DGDGT_inv, (size_t) configData->numThrusters, (size_t) 6-numZeroes, forceTorque_B, force_B);

    /* Find the minimum force */
    double min_force = force_B[0];
    for(uint32_t i = 1; i < configData->numThrusters; i++) {
        if (force_B[i] < min_force){
            min_force = force_B[i];
        }
    }

    /* Subtract the minimum force */
    for(uint32_t i = 0; i < configData->numThrusters; i++) {
        forceSubtracted_B[i] = force_B[i] - min_force;
    }

    /* Write to the output messages */
    vCopy(forceSubtracted_B, configData->numThrusters, thrForceCmdOutMsgBuffer.thrForce);
    THRArrayCmdForceMsg_C_write(&thrForceCmdOutMsgBuffer, &configData->thrForceCmdOutMsg, moduleID, callTime);
}

