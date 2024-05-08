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
    rateServoFullNonlinear Module
 
 */

#include "fswAlgorithms/attControl/rateServoFullNonlinear/rateServoFullNonlinear.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include "fswAlgorithms/fswUtilities/fswDefinitions.h"
#include "architecture/utilities/astroConstants.h"

#include <string.h>
#include <math.h>

/*! selfInit method
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_rateServoFullNonlinear(rateServoFullNonlinearConfig *configData, int64_t moduleID)
{
    CmdTorqueBodyMsg_C_init(&configData->cmdTorqueOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the servo rate control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_rateServoFullNonlinear(rateServoFullNonlinearConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Read the input messages */
    int i;
    VehicleConfigMsgPayload sc;

    /* make sure option msg connections are correctly done */
    if (RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
            _bskLog(configData->bskLogger, BSK_ERROR,"The rwSpeedsInMsg wasn't connected while rwParamsInMsg was connected.");
        }
    }

    // check if essential messages are connected
    if (!AttGuidMsg_C_isLinked(&configData->guidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rateServoFullNonlinear.guidInMsg wasn't connected.");
    }

    if (!VehicleConfigMsg_C_isLinked(&configData->vehConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rateServoFullNonlinear.vehConfigInMsg wasn't connected.");
    }

    if (!RateCmdMsg_C_isLinked(&configData->rateSteeringInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: rateServoFullNonlinear.rateSteeringInMsg wasn't connected.");
    }


    sc = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);
    for (i=0; i < 9; i++){
        configData->ISCPntB_B[i] = sc.ISCPntB_B[i];
    };
    
    configData->rwConfigParams.numRW = 0;
    if (RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);
    }
    
    /* Reset the integral measure of the rate tracking error */
    v3SetZero(configData->z);

    /* Reset the prior time flag state.
     If zero, control time step not evaluated on the first function call */
    configData->priorTime = 0;

}

/*! This method takes and rate errors relative to the Reference frame, as well as
    the reference frame angular rates and acceleration, and computes the required control torque Lr.
 @return void
 @param configData The configuration data associated with the servo rate control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_rateServoFullNonlinear(rateServoFullNonlinearConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    AttGuidMsgPayload   guidCmd;                    /*!< Guidance input Message */
    RWSpeedMsgPayload   wheelSpeeds;                /*!< Reaction wheel speed estimates input message */
    RWAvailabilityMsgPayload wheelsAvailability;    /*!< Reaction wheel availability input message */
    RateCmdMsgPayload   rateGuid;                   /*!< rate steering law message input message */
    CmdTorqueBodyMsgPayload controlOut;             /*!< commanded torque output message */

    double              dt;                 /* [s] control update period */
    
    double              Lr[3];              /* required control torque vector [Nm] */
    double              omega_BastN_B[3];   /* angular velocity of B^ast relative to inertial N, in body frame components */
    double              omega_BBast_B[3];   /* angular velocity tracking error between actual  body frame B and desired B^ast frame */
    double              omega_BN_B[3];      /* angular rate of the body B relative to inertial N, in body frame compononents */
    double              *wheelGs;           /* Reaction wheel spin axis pointer */
    /* Temporary variables */
    double              v3_1[3];
    double              v3_2[3];
    double              v3_3[3];
    double              v3_4[3];
    double              v3_5[3];
    double              v3_6[3];
    double              v3_7[3];
    int                 i;
    double              intLimCheck;
        
    /*! - zero the output message */
    controlOut = CmdTorqueBodyMsg_C_zeroMsgPayload();
    
    /*! - compute control update time */
    if (configData->priorTime == 0) {
        dt = 0.0;
    } else {
        dt = (callTime - configData->priorTime) * NANO2SEC;
    }
    configData->priorTime = callTime;

    /*! - Zero and read the dynamic input messages */
    guidCmd = AttGuidMsg_C_read(&configData->guidInMsg);
    rateGuid = RateCmdMsg_C_read(&configData->rateSteeringInMsg);


    wheelSpeeds = RWSpeedMsg_C_zeroMsgPayload();
    wheelsAvailability = RWAvailabilityMsg_C_zeroMsgPayload();  // wheelAvailability set to 0 (AVAILABLE) by default
    if(configData->rwConfigParams.numRW > 0) {
        wheelSpeeds = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);
        if (RWAvailabilityMsg_C_isLinked(&configData->rwAvailInMsg)) {
            wheelsAvailability = RWAvailabilityMsg_C_read(&configData->rwAvailInMsg);
        }
    }
    
    /*! - compute body rate */
    v3Add(guidCmd.omega_BR_B, guidCmd.omega_RN_B, omega_BN_B);

    /*! - compute the rate tracking error */
    v3Add(rateGuid.omega_BastR_B, guidCmd.omega_RN_B, omega_BastN_B);
    v3Subtract(omega_BN_B, omega_BastN_B, omega_BBast_B);

    /*! - integrate rate tracking error  */
    if (configData->Ki > 0) {   /* check if integral feedback is turned on  */
        v3Scale(dt, omega_BBast_B, v3_1);
        v3Add(v3_1, configData->z, configData->z);             /* z = integral(del_omega) */
        for (i=0;i<3;i++) {
            intLimCheck = fabs(configData->z[i]);
            if (intLimCheck > configData->integralLimit) {
                configData->z[i] *= configData->integralLimit/intLimCheck;
            }
        }
    } else {
        /* integral feedback is turned off through a negative gain setting */
        v3SetZero(configData->z);
    }

    /*! - evaluate required attitude control torque Lr */
    v3Scale(configData->P, omega_BBast_B, Lr);              /* +P delta_omega */
    v3Scale(configData->Ki, configData->z, v3_2);
    v3Add(v3_2, Lr, Lr);                                      /* +Ki*z */

    /* Lr += - omega_BastN x ([I]omega + [Gs]h_s) */
    m33MultV3(RECAST3X3 configData->ISCPntB_B, omega_BN_B, v3_3);
    for(i = 0; i < configData->rwConfigParams.numRW; i++)
    {
        if (wheelsAvailability.wheelAvailability[i] == AVAILABLE){ /* check if wheel is available */
            wheelGs = &(configData->rwConfigParams.GsMatrix_B[i*3]);
            v3Scale(configData->rwConfigParams.JsList[i] * (v3Dot(omega_BN_B, wheelGs) + wheelSpeeds.wheelSpeeds[i]),
                    wheelGs, v3_4);
            v3Add(v3_4, v3_3, v3_3);
        }
    }
    v3Cross(omega_BastN_B, v3_3, v3_4);
    v3Subtract(Lr, v3_4, Lr);
    
    /* Lr +=  - [I](d(omega_B^ast/R)/dt + d(omega_r)/dt - omega x omega_r) */
    v3Cross(omega_BN_B, guidCmd.omega_RN_B, v3_5);
    v3Subtract(guidCmd.domega_RN_B, v3_5, v3_6);
    v3Add(v3_6, rateGuid.omegap_BastR_B, v3_6);
    m33MultV3(RECAST3X3 configData->ISCPntB_B, v3_6, v3_7);
    v3Subtract(Lr, v3_7, Lr);
    
    /* Add external torque: Lr += L */
    v3Add(configData->knownTorquePntB_B, Lr, Lr);
    
    /* Change sign to compute the net positive control torque onto the spacecraft */
    v3Scale(-1.0, Lr, Lr);
    
    /*! - Set output message and pass it to the message bus */
    v3Copy(Lr, controlOut.torqueRequestBody);
    CmdTorqueBodyMsg_C_write(&controlOut, &configData->cmdTorqueOutMsg, moduleID, callTime);

    return;
}

