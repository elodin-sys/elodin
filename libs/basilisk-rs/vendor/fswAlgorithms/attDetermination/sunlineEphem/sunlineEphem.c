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

#include "sunlineEphem.h"
#include <string.h>
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"

/*!
 \verbatim embed:rst
    This method sets up the module output message of type :ref:`NavAttMsgPayload`
 \endverbatim
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_sunlineEphem(sunlineEphemConfig *configData, int64_t moduleID)
{
    NavAttMsg_C_init(&configData->navStateOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_sunlineEphem(sunlineEphemConfig *configData, uint64_t callTime, int64_t moduleID)
{
    
}

/*! Updates the sun heading based on ephemeris data. Returns the heading as a unit vector in the body frame.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_sunlineEphem(sunlineEphemConfig *configData, uint64_t callTime, int64_t moduleID)
{
    double r_SB_N[3];               /* [m] difference between the sun and spacecrat in the inertial frame (unit length) */
    double r_SB_N_hat[3];           /* [m] difference between the sun and spacecrat in the inertial frame (unit length) */
    double r_SB_B_hat[3];           /* [m] difference between the sun and spacecrat in the body frame (of unit length) */
    double BN[3][3];                /* [-] direction cosine matrix used to rotate the inertial frame to body frame */
    NavAttMsgPayload outputSunline;     /* [-] Output sunline estimate data */
    EphemerisMsgPayload sunEphemBuffer; /* [-] Input sun ephemeris data */
    NavTransMsgPayload scTransBuffer;   /* [-] Input spacecraft position data */
    NavAttMsgPayload scAttBuffer;       /* [-] Input spacecraft attitude data */
    
    // check if the required input messages are included
    if (!EphemerisMsg_C_isLinked(&configData->sunPositionInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineEphem.sunPositionInMsg wasn't connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->scPositionInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineEphem.scPositionInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->scAttitudeInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineEphem.scAttitudeInMsg wasn't connected.");
    }

    /*! - Read the input messages */
    outputSunline = NavAttMsg_C_zeroMsgPayload();
    sunEphemBuffer = EphemerisMsg_C_read(&configData->sunPositionInMsg);
    scTransBuffer = NavTransMsg_C_read(&configData->scPositionInMsg);
    scAttBuffer = NavAttMsg_C_read(&configData->scAttitudeInMsg);

    /*! - Calculate Sunline Heading from Ephemeris Data*/
    v3Subtract(sunEphemBuffer.r_BdyZero_N, scTransBuffer.r_BN_N, r_SB_N);
    v3Normalize(r_SB_N, r_SB_N_hat);
    MRP2C(scAttBuffer.sigma_BN, BN);
    m33MultV3(BN, r_SB_N_hat, r_SB_B_hat);
    v3Normalize(r_SB_B_hat, r_SB_B_hat);
    
    /*! - store the output message*/
    v3Copy(r_SB_B_hat, outputSunline.vehSunPntBdy);
    NavAttMsg_C_write(&outputSunline, &configData->navStateOutMsg, moduleID, callTime);
    return;
}
