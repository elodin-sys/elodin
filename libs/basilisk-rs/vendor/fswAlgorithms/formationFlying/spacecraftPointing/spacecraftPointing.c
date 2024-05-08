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

#include <string.h>
#include <math.h>
#include "fswAlgorithms/formationFlying/spacecraftPointing/spacecraftPointing.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/astroConstants.h"

/*! This method initializes the configData for the spacecraft pointing module
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the spacecraft pointing module
 @param moduleID The Basilisk module identifier
 */
void SelfInit_spacecraftPointing(spacecraftPointingConfig *configData, int64_t moduleID)
{
    AttRefMsg_C_init(&configData->attReferenceOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the pointing module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Reset_spacecraftPointing(spacecraftPointingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!NavTransMsg_C_isLinked(&configData->chiefPositionInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: spacecraftPointing.chiefPositionInMsg wasn't connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->deputyPositionInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: spacecraftPointing.deputyPositionInMsg wasn't connected.");
    }

    /* Build a coordinate system around the vector within the body frame that points towards the antenna and write the orientation
     of the B-frame with respect to the A-frame. */
    double dcm_AB[3][3];                            /*!< ---  dcm [AB] */
    double temp_z[3] = {0.0, 0.0, 1.0};             /*!< ---  z-axis used for cross-product */
    double temp_y[3] = {0.0, 1.0, 0.0};             /*!< ---  y-axis used for cross-product */
    double A_y_B[3];                                /*!< ---  y-axis of A-frame expressed in B-frame components */
    double A_z_B[3];                                /*!< ---  z-axis of A-frame expresses in B-frame components */
    double sigma_AB[3];                             /*!< ---  MRP of A-frame with respect to B-frame */
    v3Normalize(configData->alignmentVector_B, dcm_AB[0]);
    v3Cross(temp_z, configData->alignmentVector_B, A_y_B);
    /* If the alignment vector aligns with the z-axis of the body frame, the cross product is performed with a temporary y-axis. */
    if (v3Norm(A_y_B) < 1e-6){
        v3Cross(configData->alignmentVector_B, temp_y, A_y_B);
    }
    v3Normalize(A_y_B, dcm_AB[1]);
    v3Cross(dcm_AB[0], dcm_AB[1], A_z_B);
    v3Normalize(A_z_B, dcm_AB[2]);
    C2MRP(dcm_AB, sigma_AB);
    v3Scale(-1, sigma_AB, configData->sigma_BA);
    
    /* Set initial values of the sigma and omega of the previous timestep to zero. */
    v3SetZero(configData->old_sigma_RN);
    v3SetZero(configData->old_omega_RN_N);
    
    /* Set the numerical error flag to zero. */
    configData->i = 0;
    
    return;
}

/*! This method takes the vector that points from the deputy spacecraft to the chief spacecraft
 and calculates the orientation, angular velocity and angular acceleration of this vector with
 respect to the inertial reference frame in inertial reference frame components and passes them to
 the attitude tracking error module, where that attitude error can be calculated.
 @return void
 @param configData The configuration data associated with the spacecraft pointing module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_spacecraftPointing(spacecraftPointingConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    NavTransMsgPayload chiefTransMsg;                   /*!< ---  Input message that consists of the position and velocity of the chief */
    NavTransMsgPayload deputyTransMsg;                  /*!< ---  Input message that consists of the position and velocity of the deputy */
    double rho_N[3];                                /*!< ---  Vector pointing from deputy to chief in inertial frame components */
    double dcm_RN[3][3];                            /*!< ---  DCM from R-frame to N-frame */
    double temp_z[3] = {0.0, 0.0, 1.0};             /*!< ---  z-axis used for cross-product */
    double temp_y[3] = {0.0, 1.0, 0.0};             /*!< ---  y-axis used for cross-product */
    double R_y_N[3];                                /*!< ---  y-axis of R-frame expressed in N-frame components */
    double R_z_N[3];                                /*!< ---  z-axis of R-frame expressed in N-frame components */
    double sigma_RN[3];                             /*!< ---  MRP of vector pointing from deputy to chief */
    double delta_sigma_RN[3];                       /*!< ---  Difference between sigma at t-1 and t */
    double old_sigma_RN_shadow[3];                  /*!< ---  shadow MRP of previous timestep */
    double delta_sigma_RN_shadow[3];                /*!< ---  Difference between shadow sigma at t-1 and t */
    double dt;                                      /*!< ---  timestep of the simulation */
    double sigma_dot_RN[3];                         /*!< ---  delta_sigma_RN divided by dt results in sigma_dot_RN */
    double old_sigma_RN_squared;                    /*!< ---  sigma_RN_squared of sigma vector of previous timestep */
    double sigma_RN_squared;                        /*!< ---  sigma_RN_squared of sigma vector of current timestep */
    double old_B_sigma_RN[3][3];                    /*!< ---  B-matrix taken with sigmas from previous timestep */
    double B_sigma_RN[3][3];                        /*!< ---  B matrix to convert sigma_dot_RN to omega_RN_N */
    double B_trans[3][3];                           /*!< ---  Transposed B matrix */
    double average_scale;                           /*!< ---  average of the scaling factor (1/((1 + sigma^2)^2)) */
    double B_sigma_RN_inv[3][3];                    /*!< ---  Inverse of B matrix */
    double omega_RN_R[3];                           /*!< ---  Angular velocity of vector pointing from deputy to chief in R-frame components */
    double sigma_NR[3];                             /*!< ---  MRP of N-frame with respect to R-frame */
    double dcm_NR[3][3];                            /*!< ---  DCM [NR] */
    double omega_RN_N[3];                           /*!< ---  Angular velocity of vector pointing from deputy to chief in N-frame components */
    double delta_omega_RN_N[3];                     /*!< ---  Difference between omega at t-1 and t */
    double domega_RN_N[3];                          /*!< ---  Angular acceleration of vector pointing from deputy to chief */
    double sigma_R1N[3];                            /*!< ---  MRP of R1-frame with respect to N-frame */

    /* read in messages */
    chiefTransMsg = NavTransMsg_C_read(&configData->chiefPositionInMsg);
    deputyTransMsg = NavTransMsg_C_read(&configData->deputyPositionInMsg);

    /* Find the vector that points from the deputy spacecraft to the chief spacecraft. */
    v3Subtract(chiefTransMsg.r_BN_N, deputyTransMsg.r_BN_N, rho_N);
    
    /* Build a coordinate system around the vector that points from the deputy to the chief and
        and determine the orientation of this R-frame with respect to the N-frame. */
    v3Normalize(rho_N, dcm_RN[0]);
    v3Cross(temp_z, dcm_RN[0], R_y_N);
    /* If the rho_N vector aligns with the x-axis of the N-frame, the cross product is performed with a temporary y-axis. */
    if (v3Norm(R_y_N) < 1e-6){
        v3Cross(dcm_RN[0], temp_y, R_y_N);
    }
    v3Normalize(R_y_N, dcm_RN[1]);
    v3Cross(dcm_RN[0], dcm_RN[1], R_z_N);
    v3Normalize(R_z_N, dcm_RN[2]);
    C2MRP(dcm_RN, sigma_RN);

    /* Determine omega_RN_N */
    /* Delta sigma is calculated and the shadow delta sigma. */
    v3Subtract(sigma_RN, configData->old_sigma_RN, delta_sigma_RN);
    MRPswitch(configData->old_sigma_RN, 0.0, old_sigma_RN_shadow);
    v3Subtract(sigma_RN, old_sigma_RN_shadow, delta_sigma_RN_shadow);

    /* Usually, the norm of delta_sigma_RN_shadow is way larger than delta_sigma_RN (and not the correct delta to take).
       However, in case an MRP switch takes place, it is necessary to use delta_sigma_RN_shadow because this one will
       give the correct delta. So the if statement below makes sure that this is done. */
    if (v3Norm(delta_sigma_RN) >= v3Norm(delta_sigma_RN_shadow)){
        v3Copy(delta_sigma_RN_shadow, delta_sigma_RN);
        v3Copy(old_sigma_RN_shadow, configData->old_sigma_RN);
        }

    /* Find the timestep of the simulation. */
    dt = (callTime - configData->priorTime) * NANO2SEC;
    configData->priorTime = callTime;
        
    /* sigma_dot_RN is calculated by dividing the difference in sigma by the timestep. */
    v3Scale((1.0/dt), delta_sigma_RN, sigma_dot_RN);
        
    /* Due to the fact that sigma_dot_RN is actually the average increase in sigma over the timeperiod between t-1 and t,
       it turned out that the bevaviour of the simulation significantly improves in case the average of the B-matrix of old_sigma_RN
       and new_sigma_RN is taken, as well as the average of 1/((1+sigma^2)^2) (see Schaub and Junkins eq. 3.163). */
    old_sigma_RN_squared = configData->old_sigma_RN[0]*configData->old_sigma_RN[0] + configData->old_sigma_RN[1]*configData->old_sigma_RN[1] + configData->old_sigma_RN[2]*configData->old_sigma_RN[2];
    sigma_RN_squared = sigma_RN[0]*sigma_RN[0] + sigma_RN[1]*sigma_RN[1] + sigma_RN[2]*sigma_RN[2];
        
    m33Set(1.0 - old_sigma_RN_squared + 2.0*configData->old_sigma_RN[0]*configData->old_sigma_RN[0], 2.0*(configData->old_sigma_RN[0]*configData->old_sigma_RN[1] - configData->old_sigma_RN[2]), 2.0*(configData->old_sigma_RN[0]*configData->old_sigma_RN[2] + configData->old_sigma_RN[1]),
        2.0*(configData->old_sigma_RN[1]*configData->old_sigma_RN[0] + configData->old_sigma_RN[2]), 1.0 - old_sigma_RN_squared + 2.0*configData->old_sigma_RN[1]*configData->old_sigma_RN[1], 2.0*(configData->old_sigma_RN[1]*configData->old_sigma_RN[2] - configData->old_sigma_RN[0]),
        2.0*(configData->old_sigma_RN[2]*configData->old_sigma_RN[0] - configData->old_sigma_RN[1]), 2.0*(configData->old_sigma_RN[2]*configData->old_sigma_RN[1] + configData->old_sigma_RN[0]), 1.0 - old_sigma_RN_squared + 2.0*configData->old_sigma_RN[2]*configData->old_sigma_RN[2],
        old_B_sigma_RN);
        
    m33Set(1.0 - sigma_RN_squared + 2.0*sigma_RN[0]*sigma_RN[0], 2.0*(sigma_RN[0]*sigma_RN[1] - sigma_RN[2]), 2.0*(sigma_RN[0]*sigma_RN[2] +         sigma_RN[1]),
        2.0*(sigma_RN[1]*sigma_RN[0] + sigma_RN[2]), 1.0 - sigma_RN_squared + 2.0*sigma_RN[1]*sigma_RN[1], 2.0*(sigma_RN[1]*sigma_RN[2] - sigma_RN[0]),
        2.0*(sigma_RN[2]*sigma_RN[0] - sigma_RN[1]), 2.0*(sigma_RN[2]*sigma_RN[1] + sigma_RN[0]), 1.0 - sigma_RN_squared + 2.0*sigma_RN[2]*sigma_RN[2],
        B_sigma_RN);
        
    /* Taking the average between entries of the old sigma matrix and the new sigma matrix. */
    m33Set((old_B_sigma_RN[0][0] + B_sigma_RN[0][0])/2, (old_B_sigma_RN[0][1] + B_sigma_RN[0][1])/2, (old_B_sigma_RN[0][2] + B_sigma_RN[0][2])/2,
           (old_B_sigma_RN[1][0] + B_sigma_RN[1][0])/2, (old_B_sigma_RN[1][1] + B_sigma_RN[1][1])/2, (old_B_sigma_RN[1][2] + B_sigma_RN[1][2])/2,
           (old_B_sigma_RN[2][0] + B_sigma_RN[2][0])/2, (old_B_sigma_RN[2][1] + B_sigma_RN[2][1])/2, (old_B_sigma_RN[2][2] + B_sigma_RN[2][2])/2,
           B_sigma_RN);
        
    /* Find the angular velocity of the R-frame with respect to the N-frame according to Schaub and Junkin's chapter about MRPs. */
    m33Transpose(B_sigma_RN, B_trans);
    average_scale = 0.5*(1.0/((1.0 + sigma_RN_squared)*(1.0 + sigma_RN_squared)) + 1.0/((1.0 + old_sigma_RN_squared)*(1.0 + old_sigma_RN_squared)));
    m33Scale(average_scale, B_trans, B_sigma_RN_inv);
    m33MultV3(B_sigma_RN_inv, sigma_dot_RN, omega_RN_R);
    v3Scale(4.0, omega_RN_R, omega_RN_R);
        
    /* Convert omega_RN_R to omega_RN_N. */
    v3Scale(-1, sigma_RN, sigma_NR);
    MRP2C(sigma_NR, dcm_NR);
    m33MultV3(dcm_NR, omega_RN_R, omega_RN_N);
    
    /* Determine domega_RN_N. */
    v3Subtract(omega_RN_N, configData->old_omega_RN_N, delta_omega_RN_N);
    v3Scale((1.0/dt), delta_omega_RN_N, domega_RN_N);
    v3Copy(omega_RN_N, configData->old_omega_RN_N);

    /* Copy the sigma_RN variable to the old_sigma_RN variable. */
    v3Copy(sigma_RN, configData->old_sigma_RN);
    
    /* Due to the numerical method used the first result from omega and the first two results of domega are incorrect.
        For this reason, these values are set to zero. Take into account that the first data point is an initialization
        datapoint. This is equal to zero for all parameters. So the actual simulation only starts after this first initialization
        datapoint. */
    if (configData->i < 2){
        v3SetZero(omega_RN_N);
    }
    if (configData->i < 3){
        v3SetZero(domega_RN_N);
        configData->i += 1;
    }
    
    /* One of the requirements for this module is that the user should be able to fill in a vector within the B-frame that points at
        the antenna. For this reason it is necessary to add the orientation of the B-frame with respect to the A-frame to the R-frame
        orientation with respect to the N-frame (add sigma_BA to sigma_RN). This results in the orientation of the R1-frame with respect
        to the N-frame (sigma_R1N). In case the spacecraft B-frame points towards the R1-frame, the A-frame will point towards the R-frame,
        which results in an antenna vector that aligns with the vector that points to the chief spacecraft. */
    addMRP(sigma_RN, configData->sigma_BA, sigma_R1N);
    v3Copy(sigma_R1N, configData->attReferenceOutBuffer.sigma_RN);
    v3Copy(omega_RN_N, configData->attReferenceOutBuffer.omega_RN_N);
    v3Copy(domega_RN_N, configData->attReferenceOutBuffer.domega_RN_N);
    
    /* write the Guidance output message */
    AttRefMsg_C_write(&configData->attReferenceOutBuffer, &configData->attReferenceOutMsg, moduleID, callTime);
    return;
}
