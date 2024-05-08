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

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include "fswAlgorithms/attGuidance/celestialTwoBodyPoint/celestialTwoBodyPoint.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"

/*! This method initializes the configData for the nominal delta-V maneuver guidance.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the celestial body guidance
 @param moduleID The ID associated with the configData
 */
void SelfInit_celestialTwoBodyPoint(celestialTwoBodyPointConfig *configData,
    int64_t moduleID)
{
    AttRefMsg_C_init(&configData->attRefOutMsg);
    return;
    
}


void Reset_celestialTwoBodyPoint(celestialTwoBodyPointConfig *configData, uint64_t callTime, int64_t moduleID)
{
    configData->secCelBodyIsLinked = EphemerisMsg_C_isLinked(&configData->secCelBodyInMsg);

    // check if required input messages have been included
    if (!NavTransMsg_C_isLinked(&configData->transNavInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: celestialTwoBodyPoint.transNavInMsg wasn't connected.");
    }
    if (!EphemerisMsg_C_isLinked(&configData->celBodyInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: celestialTwoBodyPoint.celBodyInMsg wasn't connected.");
    }

    return;
}

/*! This method takes the spacecraft and points a specified axis at a named
 celestial body specified in the configuration data.  It generates the
 commanded attitude and assumes that the control errors are computed
 downstream.
 @return void
 @param configData The configuration data associated with the celestial body guidance
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_celestialTwoBodyPoint(celestialTwoBodyPointConfig *configData,
                                  uint64_t callTime, int64_t moduleID)
{
    /*! - Parse the input messages */
    parseInputMessages(configData, moduleID);
    /*! - Compute the pointing requirements */
    computeCelestialTwoBodyPoint(configData, callTime);
    /*! - Write the output message */
    AttRefMsg_C_write(&configData->attRefOut, &configData->attRefOutMsg, moduleID, callTime);
}

/*! This method takes the navigation translational info as well as the spice data of the
 primary celestial body and, if applicable, the second one, and computes the relative state vectors
 necessary to create the restricted 2-body pointing reference frame.
 @return void
 @param configData The configuration data associated with the celestial body guidance
 @param moduleID The ID associated with the configData
 */
void parseInputMessages(celestialTwoBodyPointConfig *configData, int64_t moduleID)
{
    NavTransMsgPayload navData;
    EphemerisMsgPayload primPlanet;
    EphemerisMsgPayload secPlanet;

    double R_P1B_N_hat[3];          /* Unit vector in the direction of r_P1 */
    double R_P2B_N_hat[3];          /* Unit vector in the direction of r_P2 */
    
    double platAngDiff;             /* Angle between r_P1 and r_P2 */
    double dotProduct;              /* Temporary scalar variable */

    // read input messages
    navData = NavTransMsg_C_read(&configData->transNavInMsg);
    primPlanet = EphemerisMsg_C_read(&configData->celBodyInMsg);


    v3Subtract(primPlanet.r_BdyZero_N, navData.r_BN_N, configData->R_P1B_N);
    v3Subtract(primPlanet.v_BdyZero_N, navData.v_BN_N, configData->v_P1B_N);
    
    v3SetZero(configData->a_P1B_N); /* accelerations of s/c relative to planets set to zero*/
    v3SetZero(configData->a_P2B_N);

    if(configData->secCelBodyIsLinked)
    {

        secPlanet = EphemerisMsg_C_read(&configData->secCelBodyInMsg);
        /*! - Compute R_P2 and v_P2 */
        v3Subtract(secPlanet.r_BdyZero_N, navData.r_BN_N, configData->R_P2B_N);
        v3Subtract(secPlanet.v_BdyZero_N, navData.v_BN_N, configData->v_P2B_N);
        v3Normalize(configData->R_P1B_N, R_P1B_N_hat);
        v3Normalize(configData->R_P2B_N, R_P2B_N_hat);
        dotProduct = v3Dot(R_P2B_N_hat, R_P1B_N_hat);
        platAngDiff = safeAcos(dotProduct);
    }
    
    /*! - Cross the P1 states to get R_P2, v_p2 and a_P2 */
    if(!configData->secCelBodyIsLinked ||
       fabs(platAngDiff) < configData->singularityThresh ||
       fabs(platAngDiff) > M_PI - configData->singularityThresh)
    {
        v3Cross(configData->R_P1B_N, configData->v_P1B_N, configData->R_P2B_N);
        v3Cross(configData->R_P1B_N, configData->a_P1B_N, configData->v_P2B_N);
        v3Cross(configData->v_P1B_N, configData->a_P1B_N, configData->a_P2B_N);
    }
}



/*! This method takes the spacecraft and points a specified axis at a named
 celestial body specified in the configuration data.  It generates the
 commanded attitude and assumes that the control errors are computed
 downstream.
 @return void
 @param configData The configuration data associated with the celestial body guidance
 @param callTime The clock time at which the function was called (nanoseconds)
 */
void computeCelestialTwoBodyPoint(celestialTwoBodyPointConfig *configData, uint64_t callTime)
{
    double temp3[3];        /* Temporary vector */
    double temp3_1[3];      /* Temporary vector 1 */
    double temp3_2[3];      /* Temporary vector 2 */
    double temp3_3[3];      /* Temporary vector 3 */
    double temp33[3][3];    /* Temporary 3x3 matrix */
    double temp33_1[3][3];  /* Temporary 3x3 matrix 1 */
    double temp33_2[3][3];  /* Temporary 3x3 matrix 2 */
    
    double R_N[3];          /* Normal vector of the plane defined by R_P1 and R_P2 */
    double v_N[3];          /* First time-derivative of R_n */
    double a_N[3];          /* Second time-derivative of R_n */
    
    double dcm_RN[3][3];    /* DCM that maps from Reference frame to the inertial */
    double r1_hat[3];       /* 1st row vector of RN */
    double r2_hat[3];       /* 2nd row vector of RN */
    double r3_hat[3];       /* 3rd row vector of RN */
    
    double dr1_hat[3];      /* r1_hat first time-derivative */
    double dr2_hat[3];      /* r2_hat first time-derivative */
    double dr3_hat[3];      /* r3_hat first time-derivative */
    double I_33[3][3];      /* Identity 3x3 matrix */
    double C1[3][3];        /* DCM used in the computation of rates and acceleration */
    double C3[3][3];        /* DCM used in the computation of rates and acceleration */
    
    double omega_RN_R[3];   /* Angular rate of the reference frame
                             wrt the inertial in ref R-frame components */
    double domega_RN_R[3];   /* Angular acceleration of the reference frame
                              wrt the inertial in ref R-frame components */
    
    double ddr1_hat[3];     /* r1_hat second time-derivative */
    double ddr2_hat[3];     /* r2_hat second time-derivative */
    double ddr3_hat[3];     /* r3_hat second time-derivative */
    
    configData->attRefOut = AttRefMsg_C_zeroMsgPayload();
    
    /* - Initial computations: R_n, v_n, a_n */
    v3Cross(configData->R_P1B_N, configData->R_P2B_N, R_N);
    v3Cross(configData->v_P1B_N, configData->R_P2B_N, temp3_1);
    v3Cross(configData->R_P1B_N, configData->v_P2B_N, temp3_2);
    v3Add(temp3_1, temp3_2, v_N); /*Eq 4 */
    v3Cross(configData->a_P1B_N, configData->R_P2B_N, temp3_1);
    v3Cross(configData->R_P1B_N, configData->a_P2B_N, temp3_2);
    v3Add(temp3_1, temp3_2, temp3_3);
    v3Cross(configData->v_P1B_N, configData->v_P2B_N, temp3);
    v3Scale(2.0, temp3, temp3);
    v3Add(temp3, temp3_3, a_N);  /*Eq 5*/
    
    /* - Reference Frame computation */
    v3Normalize(configData->R_P1B_N, r1_hat); /* Eq 9a*/
    v3Normalize(R_N, r3_hat); /* Eq 9c */
    v3Cross(r3_hat, r1_hat, r2_hat); /* Eq 9b */
    v3Normalize(r2_hat, r2_hat);
    v3Copy(r1_hat, dcm_RN[0]);
    v3Copy(r2_hat, dcm_RN[1]);
    v3Copy(r3_hat, dcm_RN[2]);
    C2MRP(dcm_RN, configData->attRefOut.sigma_RN);
    
    /* - Reference base-vectors first time-derivative */
    m33SetIdentity(I_33);
    
    v3OuterProduct(r1_hat, r1_hat, temp33);
    m33Subtract(I_33, temp33, C1);
    m33MultV3(C1, configData->v_P1B_N, temp3);
    v3Scale(1.0 / v3Norm(configData->R_P1B_N), temp3, dr1_hat); /* Eq 11a*/
    
    v3OuterProduct(r3_hat, r3_hat, temp33);
    m33Subtract(I_33, temp33, C3);
    m33MultV3(C3, v_N, temp3);
    v3Scale(1.0 / v3Norm(R_N), temp3, dr3_hat); /* Eq 11b*/
    
    v3Cross(dr3_hat, r1_hat, temp3_1);
    v3Cross(r3_hat, dr1_hat, temp3_2);
    v3Add(temp3_1, temp3_2, dr2_hat); /* Eq 11c*/
    
    /* - Angular velocity computation */
    omega_RN_R[0] = v3Dot(r3_hat, dr2_hat);
    omega_RN_R[1]= v3Dot(r1_hat, dr3_hat);
    omega_RN_R[2] = v3Dot(r2_hat, dr1_hat);
    m33tMultV3(dcm_RN, omega_RN_R, configData->attRefOut.omega_RN_N);
    
    /* - Reference base-vectors second time-derivative */
    m33MultV3(C1, configData->a_P1B_N, temp3_1);
    v3OuterProduct(dr1_hat, r1_hat, temp33_1);
    m33Scale(2.0, temp33_1, temp33_1);
    v3OuterProduct(r1_hat, dr1_hat, temp33_2);
    m33Add(temp33_1, temp33_2, temp33);
    m33MultV3(temp33, configData->v_P1B_N, temp3_2);
    v3Subtract(temp3_1, temp3_2, temp3);
    v3Scale(1.0 / v3Norm(configData->R_P1B_N), temp3, ddr1_hat); /* Eq 13a*/
    
    m33MultV3(C3, a_N, temp3_1);
    v3OuterProduct(dr3_hat, r3_hat, temp33_1);
    m33Scale(2.0, temp33_1, temp33_1);
    v3OuterProduct(r3_hat, dr3_hat, temp33_2);
    m33Add(temp33_1, temp33_2, temp33);
    m33MultV3(temp33, v_N, temp3_2);
    v3Subtract(temp3_1, temp3_2, temp3);
    v3Scale(1.0 / v3Norm(R_N), temp3, ddr3_hat); /* Eq 13b*/
    
    v3Cross(ddr3_hat, r1_hat, temp3_1);
    v3Cross(r3_hat, ddr1_hat, temp3_2);
    v3Add(temp3_1, temp3_2, temp3_3);
    v3Cross(dr3_hat, dr1_hat, temp3);
    v3Scale(2.0, temp3, temp3);
    v3Add(temp3, temp3_3, ddr2_hat); /* Eq 13c*/
    
    /* - Angular acceleration computation */
    domega_RN_R[0] = v3Dot(dr3_hat, dr2_hat) + v3Dot(r3_hat, ddr2_hat) - v3Dot(omega_RN_R, dr1_hat);
    domega_RN_R[1] = v3Dot(dr1_hat, dr3_hat) + v3Dot(r1_hat, ddr3_hat) - v3Dot(omega_RN_R, dr2_hat);
    domega_RN_R[2] = v3Dot(dr2_hat, dr1_hat) + v3Dot(r2_hat, ddr1_hat) - v3Dot(omega_RN_R, dr3_hat);
    m33tMultV3(dcm_RN, domega_RN_R, configData->attRefOut.domega_RN_N);
    
    return;
}

