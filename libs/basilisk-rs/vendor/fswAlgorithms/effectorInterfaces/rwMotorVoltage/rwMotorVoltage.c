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
/*
    FSW MODULE: RW motor voltage command
 
 */

#include "fswAlgorithms/effectorInterfaces/rwMotorVoltage/rwMotorVoltage.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include <string.h>


/*! This method initializes the configData for this module.
 It creates the output message.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The ID associated with the configData
 */
void SelfInit_rwMotorVoltage(rwMotorVoltageConfig *configData, int64_t moduleID)
{
    ArrayMotorVoltageMsg_C_init(&configData->voltageOutMsg);
}


/*! This method performs a reset of the module as far as closed loop control is concerned.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime Sim time in nanos
 @param moduleID The ID associated with the configData
 */
void Reset_rwMotorVoltage(rwMotorVoltageConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwMotorVoltage.rwParamsInMsg wasn't connected.");
    }

    /*! - Read static RW config data message and store it in module variables*/
    configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);

    /* reset variables */
    memset(configData->rwSpeedOld, 0, sizeof(double)*MAX_EFF_CNT);
    configData->resetFlag = BOOL_TRUE;

    /* Reset the prior time flag state.
     If zero, control time step not evaluated on the first function call */
    configData->priorTime = 0;
}

/*! Update performs the torque to voltage conversion. If a wheel speed message was provided, it also does closed loop control of the voltage sent. It then writes the voltage message.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_rwMotorVoltage(rwMotorVoltageConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /* - Read the input messages */
//    double              torqueCmd[MAX_EFF_CNT];     /*!< [Nm]   copy of RW motor torque input vector */
    ArrayMotorTorqueMsgPayload torqueCmd;           /*!< copy of RW motor torque input message*/
    ArrayMotorVoltageMsgPayload voltageOut;            /*!< -- copy of the output message */

    voltageOut = ArrayMotorVoltageMsg_C_zeroMsgPayload();

    // check if the required input messages are included
    if (!ArrayMotorTorqueMsg_C_isLinked(&configData->torqueInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rwMotorVoltage.torqueInMsg wasn't connected.");
    }

    torqueCmd = ArrayMotorTorqueMsg_C_read(&configData->torqueInMsg);

    RWSpeedMsgPayload  rwSpeed;                    /*!< [r/s] Reaction wheel speed estimates */
    rwSpeed = RWSpeedMsg_C_zeroMsgPayload();
    if(RWSpeedMsg_C_isLinked(&configData->rwSpeedInMsg)) {
        rwSpeed = RWSpeedMsg_C_read(&configData->rwSpeedInMsg);
    }
    RWAvailabilityMsgPayload  rwAvailability;
    rwAvailability = RWAvailabilityMsg_C_zeroMsgPayload(); // wheelAvailability set to 0 (AVAILABLE) by default
    if (RWAvailabilityMsg_C_isLinked(&configData->rwAvailInMsg)){
        rwAvailability = RWAvailabilityMsg_C_read(&configData->rwAvailInMsg);
    }

    /* zero the output voltage vector */
    double  voltage[MAX_EFF_CNT];       /*!< [V]   RW voltage output commands */
    memset(voltage, 0, sizeof(double)*MAX_EFF_CNT);

    /* if the torque closed-loop is on, evaluate the feedback term */
    if (RWSpeedMsg_C_isLinked(&configData->rwSpeedInMsg)) {
        /* make sure the clock didn't just initialize, or the module was recently reset */
        if (configData->priorTime != 0) {
            double dt = (callTime - configData->priorTime) * NANO2SEC; /*!< [s]   control update period */
            double              OmegaDot[MAX_EFF_CNT];     /*!< [r/s^2] RW angular acceleration */
            for (int i=0; i<configData->rwConfigParams.numRW; i++) {
                if (rwAvailability.wheelAvailability[i] == AVAILABLE && configData->resetFlag == BOOL_FALSE) {
                    OmegaDot[i] = (rwSpeed.wheelSpeeds[i] - configData->rwSpeedOld[i])/dt;
                    torqueCmd.motorTorque[i] -= configData->K * (configData->rwConfigParams.JsList[i] * OmegaDot[i] - torqueCmd.motorTorque[i]);
                }
                configData->rwSpeedOld[i] = rwSpeed.wheelSpeeds[i];
            }
            configData->resetFlag = BOOL_FALSE;
        }
        configData->priorTime = callTime;
    }

    /* evaluate the feedforward mapping of torque into voltage */
    for (int i=0; i<configData->rwConfigParams.numRW; i++) {
        if (rwAvailability.wheelAvailability[i] == AVAILABLE) {
            voltage[i] = (configData->VMax - configData->VMin)/configData->rwConfigParams.uMax[i]
                        * torqueCmd.motorTorque[i];
            if (voltage[i]>0.0) voltage[i] += configData->VMin;
            if (voltage[i]<0.0) voltage[i] -= configData->VMin;
        }
    }

    /* check for voltage saturation */
    for (int i=0; i<configData->rwConfigParams.numRW; i++) {
        if (voltage[i] > configData->VMax) {
            voltage[i] = configData->VMax;
        }
        if (voltage[i] < -configData->VMax) {
            voltage[i] = -configData->VMax;
        }
        voltageOut.voltage[i] = voltage[i];
    }

    /*
     store the output message 
     */
    ArrayMotorVoltageMsg_C_write(&voltageOut, &configData->voltageOutMsg, moduleID, callTime);

    return;
}
