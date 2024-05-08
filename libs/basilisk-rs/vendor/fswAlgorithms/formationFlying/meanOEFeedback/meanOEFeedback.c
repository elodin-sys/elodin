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

#include "meanOEFeedback.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/rigidBodyKinematics.h"

static void calc_LyapunovFeedback(meanOEFeedbackConfig *configData, NavTransMsgPayload chiefTransMsg,
                                  NavTransMsgPayload deputyTransMsg, CmdForceInertialMsgPayload *forceMsg);
static void calc_B_cl(double mu, classicElements oe_cl, double B[6][3]);
static void calc_B_eq(double mu, equinoctialElements oe_eq, double B[6][3]);
static double adjust_range(double lower, double upper, double angle);

/*! This method initializes the configData for this module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The Basilisk module identifier
 */
void SelfInit_meanOEFeedback(meanOEFeedbackConfig *configData, int64_t moduleID) {
    CmdForceInertialMsg_C_init(&configData->forceOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.  The local copy of the
 message output buffer should be cleared.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Reset_meanOEFeedback(meanOEFeedbackConfig *configData, uint64_t callTime, int64_t moduleID) {
    // check if the required input messages are included
    if (!NavTransMsg_C_isLinked(&configData->chiefTransInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: meanOEFeedback.chiefTransInMsg wasn't connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->deputyTransInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: meanOEFeedback.deputyTransInMsg wasn't connected.");
    }

    if (configData->mu <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in meanOEFeedback: mu must be set to a positive value.");
    }
    if (configData->req <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in meanOEFeedback: req must be set to a positive value.");
    }
    if (configData->J2 <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in meanOEFeedback: J2 must be set to a positive value.");
    }

    return;
}

/*! Add a description of what this main Update() routine does for this module
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_meanOEFeedback(meanOEFeedbackConfig *configData, uint64_t callTime, int64_t moduleID) {
    // in
    NavTransMsgPayload chiefTransMsg;
    NavTransMsgPayload deputyTransMsg;
    // out
    CmdForceInertialMsgPayload forceMsg;

    /*! - Read the input messages */
    chiefTransMsg = NavTransMsg_C_read(&configData->chiefTransInMsg);
    deputyTransMsg = NavTransMsg_C_read(&configData->deputyTransInMsg);

    /*! - write the module output message */
    forceMsg = CmdForceInertialMsg_C_zeroMsgPayload();
    calc_LyapunovFeedback(configData, chiefTransMsg, deputyTransMsg, &forceMsg);

    CmdForceInertialMsg_C_write(&forceMsg, &configData->forceOutMsg, moduleID, callTime);
    return;
}

/*! This function calculates Lyapunov Feedback Control output based on current orbital element difference
 and target orbital element difference. Mean orbital elements are used.
 @return void
 @param configData The configuration data associated with the module
 @param chiefTransMsg Chief's position and velocity
 @param deputyTransMsg Deputy's position and velocity
 @param forceMsg force output (3-axis)
 */
static void calc_LyapunovFeedback(meanOEFeedbackConfig *configData, NavTransMsgPayload chiefTransMsg,
                                  NavTransMsgPayload deputyTransMsg, CmdForceInertialMsgPayload *forceMsg) {
    // position&velocity to osculating classic orbital elements
    classicElements oe_cl_osc_c, oe_cl_osc_d;
    rv2elem(configData->mu, chiefTransMsg.r_BN_N, chiefTransMsg.v_BN_N, &oe_cl_osc_c);
    rv2elem(configData->mu, deputyTransMsg.r_BN_N, deputyTransMsg.v_BN_N, &oe_cl_osc_d);
    // osculating classic oe to mean classic oe
    classicElements oe_cl_mean_c, oe_cl_mean_d;
    clMeanOscMap(configData->req, configData->J2, &oe_cl_osc_c, &oe_cl_mean_c, -1);
    clMeanOscMap(configData->req, configData->J2, &oe_cl_osc_d, &oe_cl_mean_d, -1);
    // calculate necessary Force in LVLH frame
    double oed[6];
    double B[6][3];
    double force_LVLH[3];
    if (configData->oeType == 0) {
        // calculate classical oed (da,de,di,dOmega,domega,dM)
        oed[0] = (oe_cl_mean_d.a - oe_cl_mean_c.a) / oe_cl_mean_c.a - configData->targetDiffOeMean[0];
        oed[1] = oe_cl_mean_d.e - oe_cl_mean_c.e - configData->targetDiffOeMean[1];
        oed[2] = oe_cl_mean_d.i - oe_cl_mean_c.i - configData->targetDiffOeMean[2];
        oed[3] = oe_cl_mean_d.Omega - oe_cl_mean_c.Omega - configData->targetDiffOeMean[3];
        oed[4] = oe_cl_mean_d.omega - oe_cl_mean_c.omega - configData->targetDiffOeMean[4];
        double E_mean_c = f2E(oe_cl_mean_c.f, oe_cl_mean_c.e);
        double M_mean_c = E2M(E_mean_c, oe_cl_mean_c.e);
        double E_mean_d = f2E(oe_cl_mean_d.f, oe_cl_mean_d.e);
        double M_mean_d = E2M(E_mean_d, oe_cl_mean_d.e);
        oed[5] = M_mean_d - M_mean_c - configData->targetDiffOeMean[5];
        oed[2] = adjust_range(-M_PI, M_PI, oed[2]);
        oed[3] = adjust_range(-M_PI, M_PI, oed[3]);
        oed[4] = adjust_range(-M_PI, M_PI, oed[4]);
        oed[5] = adjust_range(-M_PI, M_PI, oed[5]);
        // calculate control matrix B
        calc_B_cl(configData->mu, oe_cl_mean_d, B);
    } else if (configData->oeType == 1) {
        // mean classic oe to mean equinoctial oe
        equinoctialElements oe_eq_mean_c, oe_eq_mean_d;
        clElem2eqElem(&oe_cl_mean_c, &oe_eq_mean_c);
        clElem2eqElem(&oe_cl_mean_d, &oe_eq_mean_d);
        // calculate equinoctial oed (da,dP1,dP2,dQ1,dQ2,dl)
        oed[0] = (oe_eq_mean_d.a - oe_eq_mean_c.a) / oe_eq_mean_c.a - configData->targetDiffOeMean[0];
        oed[1] = oe_eq_mean_d.P1 - oe_eq_mean_c.P1 - configData->targetDiffOeMean[1];
        oed[2] = oe_eq_mean_d.P2 - oe_eq_mean_c.P2 - configData->targetDiffOeMean[2];
        oed[3] = oe_eq_mean_d.Q1 - oe_eq_mean_c.Q1 - configData->targetDiffOeMean[3];
        oed[4] = oe_eq_mean_d.Q2 - oe_eq_mean_c.Q2 - configData->targetDiffOeMean[4];
        oed[5] = oe_eq_mean_d.l - oe_eq_mean_c.l - configData->targetDiffOeMean[5];
        oed[5] = adjust_range(-M_PI, M_PI, oed[5]);
        // calculate control matrix B
        calc_B_eq(configData->mu, oe_eq_mean_d, B);
    }
    // calculate Lyapunov Feedback Control
    double K_oed[6];
    m66MultV6(RECAST6X6 configData->K, oed, K_oed);
    mtMultV(B, 6, 3, K_oed, force_LVLH);
    v3Scale(-1, force_LVLH, force_LVLH);
    // convert force to Inertial frame
    double dcm_RN[3][3];
    double h[3];
    v3Cross(deputyTransMsg.r_BN_N, deputyTransMsg.v_BN_N, h);
    v3Normalize(deputyTransMsg.r_BN_N, dcm_RN[0]);
    v3Normalize(h, dcm_RN[2]);
    v3Cross(dcm_RN[2], dcm_RN[0], dcm_RN[1]);
    m33tMultV3(dcm_RN, force_LVLH, forceMsg->forceRequestInertial);
    return;
}

/*! This function calculates Control Matrix (often called B matrix) derived from Gauss' Planetary Equation.
 Especially, this function assumes using classic orbital elements.
 The B matrix description is provided in
 "Analytical Mechanics of Space Systems by H. Schaub and J. L. Junkins"
 @return void
 @param mu
 @param oe_cl nonsingular orbital elements
 @param B
 */
static void calc_B_cl(double mu, classicElements oe_cl, double B[6][3]) {
    // define parameters necessary to calculate Bmatrix
    double a = oe_cl.a;
    double e = oe_cl.e;
    double i = oe_cl.i;
//    double Omega = oe_cl.Omega;
    double omega = oe_cl.omega;
    double f = oe_cl.f;
    double theta = omega + f;
    double eta = sqrt(1 - e * e);
    double b = a * sqrt(1 - e * e);
    double n = sqrt(mu / pow(a, 3));
    double h = n * a * b;
    double p = a * (1 - e * e);
    double r = p / (1 + e * cos(f));

    // sabstitute into Bmatrix
    B[0][0] = 2.0 * pow(a, 2) * e * sin(f) / h / a;  // nomalization
    B[0][1] = 2.0 * pow(a, 2) * p / (h * r) / a;     // nomalization
    B[0][2] = 0;

    B[1][0] = p * sin(f) / h;
    B[1][1] = ((p + r) * cos(f) + r * e) / h;
    B[1][2] = 0;

    B[2][0] = 0;
    B[2][1] = 0;
    B[2][2] = r * cos(theta) / h;

    B[3][0] = 0;
    B[3][1] = 0;
    B[3][2] = r * sin(theta) / (h * sin(i));

    B[4][0] = -p * cos(f) / (h * e);
    B[4][1] = (p + r) * sin(f) / (h * e);
    B[4][2] = -r * sin(theta) * cos(i) / (h * sin(i));

    B[5][0] = eta * (p * cos(f) - 2 * r * e) / (h * e);
    B[5][1] = -eta * (p + r) * sin(f) / (h * e);
    B[5][2] = 0;
    return;
}

/*! This function calculates Control Matrix (often called B matrix) derived from Gauss' Planetary Equation.
 Especially, this function assumes using nonsingular orbital elements, which help to avoid singularity.
 The B matrix description is provided in
 "Naasz, B. J., Karlgaard, C. D., & Hall, C. D. (2002). Application of several control techniques for
 the ionospheric observation nanosatellite formation."
 Be careful, our definition of equinoctial orbital elements are different from the one used in this paper.
 @return void
 @param mu
 @param oe_eq nonsingular orbital elements
 @param B
 */
static void calc_B_eq(double mu, equinoctialElements oe_eq, double B[6][3]) {
    // define parameters necessary to calculate Bmatrix
    double a = oe_eq.a;
    double P1 = oe_eq.P1;
    double P2 = oe_eq.P2;
    double Q1 = oe_eq.Q1;
    double Q2 = oe_eq.Q2;
    double L = oe_eq.L;
    double b = a * sqrt(1 - P1 * P1 - P2 * P2);
    double n = sqrt(mu / pow(a, 3));
    double h = n * a * b;
    double p_r = 1 + P1 * sin(L) + P2 * cos(L);
    double r_h = h / (mu * p_r);
    double r = r_h * h;
    double p = p_r * r;

    // sabstitute into Bmatrix
    B[0][0] = 2.0 * pow(a, 2) / h * (P2 * sin(L) - P1 * cos(L)) / a;  // nomalization
    B[0][1] = 2.0 * pow(a, 2) * p_r / h / a;                          // nomalization
    B[0][2] = 0;

    B[1][0] = -p * cos(L) / h;
    B[1][1] = 1.0 / h * (r * P1 + (r + p) * sin(L));
    B[1][2] = r_h * P2 * (Q2 * sin(L) - Q1 * cos(L));

    B[2][0] = p * sin(L) / h;
    B[2][1] = 1.0 / h * (r * P2 + (r + p) * cos(L));
    B[2][2] = r_h * P1 * (Q1 * cos(L) - Q2 * sin(L));

    B[3][0] = 0;
    B[3][1] = 0;
    B[3][2] = 1.0 / 2.0 * r_h * (1 + Q1 * Q1 + Q2 * Q2) * sin(L);

    B[4][0] = 0;
    B[4][1] = 0;
    B[4][2] = 1.0 / 2.0 * r_h * (1 + Q1 * Q1 + Q2 * Q2) * cos(L);

    B[5][0] = -p * a / (h * (a + b)) * ((P1 * sin(L) + P2 * cos(L)) + 2.0 * b / a);
    B[5][1] = r_h * a * (1.0 + p_r) / (a + b) * (P2 * sin(L) - P1 * cos(L));
    B[5][2] = r_h * (Q2 * sin(L) - Q1 * cos(L));
    return;
}

/*! This function is used to adjust a certain value in a certain range between lower threshold and upper threshold.
 This function is particularily used to adjsut angles used in orbital motions such as True Anomaly, Mean Anomaly, and so on.
 @return double
 @param lower lower threshold
 @param upper upper threshold
 @param angle an angle which you want to be between lower and upper
*/
static double adjust_range(double lower, double upper, double angle) {
    if (upper < lower) {
        printf("illegal parameters\n");
        return -1;
    }
    double width = upper - lower;
    double adjusted_angle = angle;
    while (adjusted_angle > upper) {
        adjusted_angle = adjusted_angle - width;
    }
    while (adjusted_angle < lower) {
        adjusted_angle = adjusted_angle + width;
    }
    return adjusted_angle;
}
