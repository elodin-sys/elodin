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
    MRP_STEERING Module
 
 */

#include "fswAlgorithms/attControl/mrpSteering/mrpSteering.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/astroConstants.h"
#include <string.h>
#include <math.h>

/*! self init method
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
*/
void SelfInit_mrpSteering(mrpSteeringConfig *configData, int64_t moduleID)
{
    RateCmdMsg_C_init(&configData->rateCmdOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the MRP steering control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Reset_mrpSteering(mrpSteeringConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check for required input message
    if (!AttGuidMsg_C_isLinked(&configData->guidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: mrpSteering.guidInMsg wasn't connected.");
    }

    return;
}

/*! This method takes the attitude and rate errors relative to the Reference frame, as well as
    the reference frame angular rates and acceleration
 @return void
 @param configData The configuration data associated with the MRP Steering attitude control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_mrpSteering(mrpSteeringConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    AttGuidMsgPayload guidCmd;              /* Guidance Message */
    RateCmdMsgPayload outMsg;               /* copy of output message */
    
    /*! - Zero message copies*/
    outMsg = RateCmdMsg_C_zeroMsgPayload();

    /*! - Read the dynamic input messages */
    guidCmd = AttGuidMsg_C_read(&configData->guidInMsg);

    /*! - evalute MRP kinematic steering law */
    MRPSteeringLaw(configData, guidCmd.sigma_BR, outMsg.omega_BastR_B, outMsg.omegap_BastR_B);

    /*! - Store the output message and pass it to the message bus */
    RateCmdMsg_C_write(&outMsg, &configData->rateCmdOutMsg, moduleID, callTime);

    return;
}

/*! This method computes the MRP Steering law.  A commanded body rate is returned given the MRP
 attitude error measure of the body relative to a reference frame.  The function returns the commanded
 body rate, as well as the body frame derivative of this rate command.
 @return void
 @param configData  The configuration data associated with this module
 @param sigma_BR    MRP attitude error of B relative to R
 @param omega_ast   Commanded body rates
 @param omega_ast_p Body frame derivative of the commanded body rates
 */
void MRPSteeringLaw(mrpSteeringConfig *configData, double sigma_BR[3], double omega_ast[3], double omega_ast_p[3])
{
    double  sigma_i;        /* ith component of sigma_B/R */
    double  B[3][3];        /* B-matrix of MRP differential kinematic equations */
    double  sigma_p[3];     /* MRP rates */
    double  value;
    int     i;

    /* Equation (18): Determine the desired steering rates  */
    for (i=0;i<3;i++) {
        sigma_i  = sigma_BR[i];
        value        = atan(M_PI_2/configData->omega_max*(configData->K1*sigma_i
                       + configData->K3*sigma_i*sigma_i*sigma_i))/M_PI_2*configData->omega_max;
        omega_ast[i] = -value;
    }
    v3SetZero(omega_ast_p);
    if (!configData->ignoreOuterLoopFeedforward) {
        /* Equation (21): Determine the body frame derivative of the steering rates */
        BmatMRP(sigma_BR, B);
        m33MultV3(B, omega_ast, sigma_p);
        v3Scale(0.25, sigma_p, sigma_p);
        for (i=0;i<3;i++) {
            sigma_i  = sigma_BR[i];
            value = (3*configData->K3*sigma_i*sigma_i + configData->K1)/
                                (pow(M_PI_2/configData->omega_max*(configData->K1*sigma_i + configData->K3*sigma_i*sigma_i*sigma_i),2) + 1);
            omega_ast_p[i] = - value*sigma_p[i];
        }
    }
    return;
}
