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


#include "torqueScheduler.h"
#include "string.h"


/*! This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_torqueScheduler(torqueSchedulerConfig *configData, int64_t moduleID)
{
    ArrayMotorTorqueMsg_C_init(&configData->motorTorqueOutMsg);
    ArrayEffectorLockMsg_C_init(&configData->effectorLockOutMsg);
}

/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_torqueScheduler(torqueSchedulerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    if (!ArrayMotorTorqueMsg_C_isLinked(&configData->motorTorque1InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "torqueScheduler.motorTorque1InMsg wasn't connected.");
    }
    if (!ArrayMotorTorqueMsg_C_isLinked(&configData->motorTorque2InMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "torqueScheduler.motorTorque2InMsg wasn't connected.");
    }

    configData->t0 = callTime;
}

/*! This method computes the control torque to the solar array drive based on a PD control law
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_torqueScheduler(torqueSchedulerConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Create and assign buffer messages */
    ArrayMotorTorqueMsgPayload  motorTorque1In  = ArrayMotorTorqueMsg_C_read(&configData->motorTorque1InMsg);
    ArrayMotorTorqueMsgPayload  motorTorque2In  = ArrayMotorTorqueMsg_C_read(&configData->motorTorque2InMsg);
    ArrayMotorTorqueMsgPayload  motorTorqueOut  = ArrayMotorTorqueMsg_C_zeroMsgPayload();
    ArrayEffectorLockMsgPayload effectorLockOut = ArrayEffectorLockMsg_C_zeroMsgPayload();

    /*! compute current time from Reset call */
    double t = ((callTime - configData->t0) * NANO2SEC);

    /*! populate output torque msg */
    motorTorqueOut.motorTorque[0] = motorTorque1In.motorTorque[0];
    motorTorqueOut.motorTorque[1] = motorTorque2In.motorTorque[0];
    
    switch (configData->lockFlag) {

        case 0:
            effectorLockOut.effectorLockFlag[0] = 0;
            effectorLockOut.effectorLockFlag[1] = 0;
            break;

        case 1:
            if (t > configData->tSwitch) {
                effectorLockOut.effectorLockFlag[0] = 1;
                effectorLockOut.effectorLockFlag[1] = 0;
            }
            else {
                effectorLockOut.effectorLockFlag[0] = 0;
                effectorLockOut.effectorLockFlag[1] = 1;
            }
            break;

        case 2:
            if (t > configData->tSwitch) {
                effectorLockOut.effectorLockFlag[0] = 0;
                effectorLockOut.effectorLockFlag[1] = 1;
            }
            else {
                effectorLockOut.effectorLockFlag[0] = 1;
                effectorLockOut.effectorLockFlag[1] = 0;
            }
            break;

        case 3:
            effectorLockOut.effectorLockFlag[0] = 1;
            effectorLockOut.effectorLockFlag[1] = 1;
            break;

        default:
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: torqueScheduler.lockFlag has to be an integer between 0 and 3.");

    }

    /* write output messages */
    ArrayMotorTorqueMsg_C_write(&motorTorqueOut, &configData->motorTorqueOutMsg, moduleID, callTime);
    ArrayEffectorLockMsg_C_write(&effectorLockOut, &configData->effectorLockOutMsg, moduleID, callTime);
}
