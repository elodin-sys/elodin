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
    mrpFeedback Module

 */

#include "fswAlgorithms/attControl/mrpFeedback/mrpFeedback.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/astroConstants.h"

#include <string.h>
#include <math.h>

/*! @brief This method sets up the module output message of type :ref:`CmdTorqueBodyMsgPayload`
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The ID associated with the configData
*/
void SelfInit_mrpFeedback(mrpFeedbackConfig *configData, int64_t moduleID)
{
    CmdTorqueBodyMsg_C_init(&configData->cmdTorqueOutMsg);
    CmdTorqueBodyMsg_C_init(&configData->intFeedbackTorqueOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param callTime The clock time at which the function was called (nanoseconds)
 @param configData The configuration data associated with the MRP steering control
 @param moduleID The module identifier
*/
void Reset_mrpFeedback(mrpFeedbackConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /* - Read the input messages */
    int i;

    /* check that optional messages are correct connected */
    if(RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
            _bskLog(configData->bskLogger, BSK_ERROR, "Error: the mrpFeedback.rwSpeedsInMsg wasn't connected while rwParamsInMsg was connected.");
        }

    }

    // check if the required message has not been connected
    if (!AttGuidMsg_C_isLinked(&configData->guidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mrpFeedback.guidInMsg wasn't connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->vehConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: the mrpFeedback.vehConfigInMsg wasn't connected.");
    }

    /*! - zero and read in vehicle configuration message */
    VehicleConfigMsgPayload sc;
    sc = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);
    /*! - copy over spacecraft inertia tensor */
    for (i=0; i < 9; i++){
        configData->ISCPntB_B[i] = sc.ISCPntB_B[i];
    };

    /*! - zero the number of RW by default */
    configData->rwConfigParams.numRW = 0;

    /*! - check if RW configuration message exists */
    if (RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        /*! - Read static RW config data message and store it in module variables*/
        configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);
    }

    /*! - Reset the integral measure of the rate tracking error */
    v3SetZero(configData->int_sigma);

    /*! - Reset the prior time flag state.
     If zero, control time step not evaluated on the first function call */
    configData->priorTime = 0;
}

/*! This method takes the attitude and rate errors relative to the Reference frame, as well as
    the reference frame angular rates and acceleration, and computes the required control torque Lr.
 @return void
 @param configData The configuration data associated with the MRP Steering attitude control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_mrpFeedback(mrpFeedbackConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    AttGuidMsgPayload      guidCmd;            /* attitude tracking error message */
    RWSpeedMsgPayload      wheelSpeeds;        /* Reaction wheel speed message */
    RWAvailabilityMsgPayload wheelsAvailability; /* Reaction wheel availability message */
    CmdTorqueBodyMsgPayload controlOut;        /* output message */
    CmdTorqueBodyMsgPayload intFeedbackOut;    /* output int feedback msg */

    double              dt;                 /* [s] control update period */
    double              Lr[3];              /* required control torque vector [Nm] */
    double              omega_BN_B[3];      /* [r/s] body angular velocity message */
    double              v3_1[3];
    double              v3_2[3];
    double              v3_3[3];
    double              v3_4[3];
    double              v3_5[3];
    double              v3_6[3];
    double              v3_7[3];
    double              v3_8[3];
    double              v3_9[3];
    double              v3_10[3];
    double              v3_11[3];
    double              v3_12[3];
    double              intCheck;           /* Check magnitude of integrated attitude error */
    int                 i;
    double              *wheelGs;           /* Reaction wheel spin axis pointer */

    /*! - zero the output message */
    controlOut = CmdTorqueBodyMsg_C_zeroMsgPayload();

    /*! - Read the attitude tracking error message */
    guidCmd = AttGuidMsg_C_read(&configData->guidInMsg);

    /*! - read in optional RW speed and availability message */
    if(configData->rwConfigParams.numRW > 0) {
        wheelSpeeds = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);
        wheelsAvailability = RWAvailabilityMsg_C_zeroMsgPayload(); /* wheelAvailability set to 0 (AVAILABLE) by default */
        if (RWAvailabilityMsg_C_isLinked(&configData->rwAvailInMsg)) {
            wheelsAvailability = RWAvailabilityMsg_C_read(&configData->rwAvailInMsg);
        }
    }

    /*! - compute control update time */
    if (configData->priorTime == 0) {
        dt = 0.0;
    } else {
        dt = (callTime - configData->priorTime) * NANO2SEC;
    }
    configData->priorTime = callTime;

    /*! - compute body rate */
    v3Add(guidCmd.omega_BR_B, guidCmd.omega_RN_B, omega_BN_B);

    /*! - evaluate integral term */
    v3SetZero(configData->z);
    if (configData->Ki > 0) {   /* check if integral feedback is turned on  */
        v3Scale(configData->K * dt, guidCmd.sigma_BR, v3_1);
        v3Add(v3_1, configData->int_sigma, configData->int_sigma);

        for (i=0;i<3;i++) {
            intCheck = fabs(configData->int_sigma[i]);
            if (intCheck > configData->integralLimit) {
                configData->int_sigma[i] *= configData->integralLimit/intCheck;
            }
        }/* keep int_sigma less than integralLimit */
        m33MultV3(RECAST3X3 configData->ISCPntB_B, guidCmd.omega_BR_B, v3_2); /* -[v3Tilde(omega_r+Ki*z)]([I]omega + [Gs]h_s) */
        v3Add(configData->int_sigma, v3_2, configData->z);
    }

    /*! - evaluate required attitude control torque Lr */
    v3Scale(configData->K, guidCmd.sigma_BR, Lr);           /* +K sigma_BR */
    v3Scale(configData->P, guidCmd.omega_BR_B,
            v3_3);                                          /* +P delta_omega */
    v3Add(v3_3, Lr, Lr);
    v3Scale(configData->Ki, configData->z, v3_4);
    v3Scale(configData->P, v3_4, v3_5);                       /* +P*Ki*z */
    v3Add(v3_5, Lr, Lr);

    /* -[v3Tilde(omega_r+Ki*z)]([I]omega + [Gs]h_s) */
    m33MultV3(RECAST3X3 configData->ISCPntB_B, omega_BN_B, v3_6);
    for(i = 0; i < configData->rwConfigParams.numRW; i++)
    {
        if (wheelsAvailability.wheelAvailability[i] == AVAILABLE){ /* check if wheel is available */
            wheelGs = &(configData->rwConfigParams.GsMatrix_B[i*3]);
            v3Scale(configData->rwConfigParams.JsList[i] * (v3Dot(omega_BN_B, wheelGs) + wheelSpeeds.wheelSpeeds[i]),
                    wheelGs, v3_7);                                 /* h_s_i */
            v3Add(v3_6, v3_7, v3_6);
        }
    }

    if (configData->controlLawType == 0) {
        v3Add(guidCmd.omega_RN_B, v3_4, v3_8);      /* v3_8 = omega_RN_B + K_I * z */
    }
    else {
        v3Copy(omega_BN_B, v3_8);                        /* v3_8 = omega_BN_B */
    }
    v3Cross(v3_8, v3_6, v3_9);
    v3Subtract(Lr, v3_9, Lr);

    v3Cross(omega_BN_B, guidCmd.omega_RN_B, v3_10);
    v3Subtract(v3_10, guidCmd.domega_RN_B, v3_11);
    m33MultV3(RECAST3X3 configData->ISCPntB_B, v3_11, v3_12);   /* +[I](-d(omega_r)/dt + omega x omega_r) */
    v3Add(v3_12, Lr, Lr);

    v3Add(configData->knownTorquePntB_B, Lr, Lr);           /* +L */
    v3Scale(-1.0, Lr, Lr);                                  /* compute the net positive control torque onto the spacecraft */


    /*! - set the output message and write it out */
    v3Copy(Lr, controlOut.torqueRequestBody);
    CmdTorqueBodyMsg_C_write(&controlOut, &configData->cmdTorqueOutMsg, moduleID, callTime);

    /*! - write the output integral feedback torque */
    v3Scale(-1.0, v3_5, intFeedbackOut.torqueRequestBody);
    CmdTorqueBodyMsg_C_write(&intFeedbackOut, &configData->intFeedbackTorqueOutMsg, moduleID, callTime);

    return;
}
