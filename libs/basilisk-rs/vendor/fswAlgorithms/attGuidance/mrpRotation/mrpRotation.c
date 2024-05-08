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
 MRP Rotation Guidance Module with a Constant Body Rate Vector
 
 */

#include "fswAlgorithms/attGuidance/mrpRotation/mrpRotation.h"
#include <string.h>
#include <math.h>
#include "fswAlgorithms/fswUtilities/fswDefinitions.h"
#include "architecture/utilities/macroDefinitions.h"

/* Support files.  Be sure to use the absolute path relative to Basilisk directory. */
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"

/*! @brief This method initializes the configData for mrpRotation model.  It creates the module
 output message.
 @return void
 @param configData The configuration data associated with the null space control
 @param moduleID The ID associated with the configData
 */
void SelfInit_mrpRotation(mrpRotationConfig *configData, int64_t moduleID)
{
    AttRefMsg_C_init(&configData->attRefOutMsg);

    return;
}


/*! @brief This resets the module to original states.
 @return void
 @param configData The configuration data associated with the null space control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_mrpRotation(mrpRotationConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!AttRefMsg_C_isLinked(&configData->attRefInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mrpRotation.attRefInMsg wasn't connected.");
    }

    configData->priorTime = 0;

    v3SetZero(configData->priorCmdSet);
    v3SetZero(configData->priorCmdRates);
     
}

/*! @brief This method takes the input attitude reference frame, and and superimposes the dynamics MRP
 scanning motion on top of this.
 @return void
 @param configData The configuration data associated with the mrpRotation module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_mrpRotation(mrpRotationConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /* - Read input messages */
    AttRefMsgPayload inputRef;                                /* [-] read in the [R_0N] input reference message */
    AttRefMsgPayload attRefOut;                               /* [-] structure for the Reference frame output data */

    /*!- read in input reference frame message */
    inputRef = AttRefMsg_C_read(&configData->attRefInMsg);

    /*! - Check if a desired attitude configuration message exists. This allows for dynamic changes to the desired MRP rotation */
    if (AttStateMsg_C_isLinked(&configData->desiredAttInMsg))
    {
        AttStateMsgPayload attStates;                         /* [-] initial [RR_0] attitude state message */

        /* - Read Raster Manager messages */
        attStates = AttStateMsg_C_read(&configData->desiredAttInMsg);

        /* - Save commanded MRP set and body rates */
        v3Copy(attStates.state, configData->cmdSet);
        v3Copy(attStates.rate, configData->cmdRates);
        /* - Check the command is new */
        checkRasterCommands(configData);
    }
    
    /*! - Compute time step to use in the integration downstream */
    computeTimeStep(configData, callTime);

    /*! - Compute output reference frame */
    attRefOut = AttRefMsg_C_zeroMsgPayload();
    computeMRPRotationReference(configData,
                                inputRef.sigma_RN,
                                inputRef.omega_RN_N,
                                inputRef.domega_RN_N,
                                &attRefOut);


    /*! - write attitude guidance reference output */
    AttRefMsg_C_write(&attRefOut, &configData->attRefOutMsg, moduleID, callTime);

    /*! - Update last time the module was called to current call time */
    configData->priorTime = callTime;
    return;
}



/*! @brief This function checks if there is a new commanded raster maneuver message available
 @return void
 @param configData The configuration data associated with the mrpRotation module
 */
void checkRasterCommands(mrpRotationConfig *configData)
{
    int32_t prevCmdActive = v3IsEqual(configData->cmdSet, configData->priorCmdSet , 1E-12)
                            && v3IsEqual(configData->cmdRates, configData->priorCmdRates , 1E-12);
    /*! - check if a new attitude reference command message content is availble */
    if (!prevCmdActive)
    {
        /*! - copy over the commanded initial MRP and rate information */
        v3Copy(configData->cmdSet, configData->mrpSet);
        v3Copy(configData->cmdRates, configData->omega_RR0_R);

        /*! - reset the prior commanded attitude state variables */
        v3Copy(configData->cmdSet, configData->priorCmdSet);
        v3Copy(configData->cmdRates, configData->priorCmdRates);
    }
}

/*! @brief This function computes control update time
 @return void
 @param configData The configuration data associated with the mrpRotation module
 @param callTime The clock time at which the function was called (nanoseconds)
*/
void computeTimeStep(mrpRotationConfig *configData, uint64_t callTime)
{
    if (configData->priorTime == 0)
    {
        configData->dt = 0.0;
    } else {
        configData->dt = (callTime - configData->priorTime)*NANO2SEC;
    }
}


/*! @brief This function computes the reference (MRP attitude Set, angular velocity and angular acceleration)
 associated with a rotation defined in terms of an initial MRP set and a constant angular velocity vector
 @return void
 @param configData The configuration data associated with the mrpRotation module
 @param sigma_R0N The input reference attitude using MRPs
 @param omega_R0N_N The input reference frame angular rate vector
 @param domega_R0N_N The input reference frame angular acceleration vector
 @param attRefOut The output message copy
 */
void computeMRPRotationReference(mrpRotationConfig *configData,
                                 double sigma_R0N[3],
                                 double omega_R0N_N[3],
                                 double domega_R0N_N[3],
                                 AttRefMsgPayload   *attRefOut)
{
    double attIncrement[3];         /* [] increment in MRP attitude coordinates  */
    double RR0[3][3];               /* [] DCM rotating from R0 to R */
    double R0N[3][3];               /* [] DCM rotating from N to R0 */
    double RN[3][3];                /* [] DCM rotating from N to R */
    double sigmaDot_RR0[3];         /* [1/s] MRP rates */
    double B[3][3];                 /* [] B matrix relating omega to MRP rates */
    double omega_RR0_N[3];          /* [r/s] angular velocity of R frame relative to input R0 frame */
    double domega_RR0_N[3];         /* [r/s^2] inertial derivative of angular velocity vector between R and R0 frames */

    /*! - Compute attitude reference frame R/N information */
    MRP2C(sigma_R0N, R0N);
    BmatMRP(configData->mrpSet, B);
    m33MultV3(B, configData->omega_RR0_R, sigmaDot_RR0);
    v3Scale(0.25, sigmaDot_RR0, sigmaDot_RR0);
    v3Scale(configData->dt, sigmaDot_RR0, attIncrement);
    v3Add(configData->mrpSet, attIncrement, configData->mrpSet);
    MRPswitch(configData->mrpSet, 1.0, configData->mrpSet);
    MRP2C(configData->mrpSet, RR0);
    m33MultM33(RR0, R0N, RN);
    C2MRP(RN, attRefOut->sigma_RN);
    
    /*! - Compute angular velocity of R/N */
    m33tMultV3(RN, configData->omega_RR0_R, omega_RR0_N);
    v3Add(omega_R0N_N, omega_RR0_N, attRefOut->omega_RN_N);
 
    /*! - Compute angular acceleration of R/N */
    v3Cross(omega_R0N_N, omega_RR0_N, domega_RR0_N);
    v3Add(domega_RR0_N, domega_R0N_N, attRefOut->domega_RN_N);
}
