/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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

#include "fswAlgorithms/orbitControl/smallBodyWaypointFeedback/smallBodyWaypointFeedback.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <cstring>
#include <math.h>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
SmallBodyWaypointFeedback::SmallBodyWaypointFeedback()
{
    this->mu_sun = 1.327124e20;
    this->o_hat_3_tilde.setZero();
    this->o_hat_3_tilde(0, 1) = -1;
    this->o_hat_3_tilde(1, 0) = 1;
    this->o_hat_1 << 1, 0, 0;
    this->I.setIdentity(3,3);
    this->C_SRP = 1.0;
    this->P_0 = 4.56e-6;
    this->rho = 0.4;
    this->prevTime = 0.0;
    return;
}

/*! Module Destructor */
SmallBodyWaypointFeedback::~SmallBodyWaypointFeedback()
{
}

/*! Initialize C-wrapped output messages */
void SmallBodyWaypointFeedback::SelfInit(){
    CmdForceBodyMsg_C_init(&this->forceOutMsgC);
}

/*! This method is used to reset the module and checks that required input messages are connect.
    @return void
*/
void SmallBodyWaypointFeedback::Reset(uint64_t CurrentSimNanos)
{
    // check that required input messages are connected
    if (!this->navTransInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyWaypointFeedback.navTransInMsg was not linked.");
    }
    if (!this->navAttInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyWaypointFeedback.navAttInMsg was not linked.");
    }
    if (!this->asteroidEphemerisInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyWaypointFeedback.asteroidEphemerisInMsg was not linked.");
    }
    if (!this->sunEphemerisInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyWaypointFeedback.sunEphemerisInMsg was not linked.");
    }

}

/*! This method reads the input messages each call of updateState
    @return void
*/
void SmallBodyWaypointFeedback::readMessages(){
    /* read in the input messages */
    navTransInMsgBuffer = this->navTransInMsg();
    navAttInMsgBuffer = this->navAttInMsg();
    asteroidEphemerisInMsgBuffer = this->asteroidEphemerisInMsg();
    sunEphemerisInMsgBuffer = this->sunEphemerisInMsg();
}

/*! This method computes the control using a Lyapunov feedback law
    @return void
*/
void SmallBodyWaypointFeedback::computeControl(uint64_t CurrentSimNanos){
    /* Get the orbital elements of the asteroid, we assume the uncertainty on the pos. and vel. of the body are low
     * enough to consider them known apriori */
    rv2elem(mu_sun, asteroidEphemerisInMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N, &oe_ast);

    /* Compute F_dot and F_ddot */
    F_dot = sqrt(mu_sun / (pow(oe_ast.a * (1 - pow(oe_ast.e, 2)), 3))) * pow(1 + (oe_ast.e) * cos(oe_ast.f), 2);
    F_ddot = -2 * (oe_ast.e) * (sqrt(mu_sun / (pow(oe_ast.a * (1 - pow(oe_ast.e, 2)), 3)))) * sin(oe_ast.f) *
             (1 + oe_ast.e * cos(oe_ast.f)) * (F_dot);

    /* Compute the hill frame DCM of the small body */
    double dcm_ON_array[3][3];
    hillFrame(asteroidEphemerisInMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N, dcm_ON_array);
    dcm_ON = cArray2EigenMatrixXd(*dcm_ON_array, 3, 3).transpose();

    /* Compute the direction of the sun from the asteroid in the small body's hill frame, assumes heliocentric frame
     * centered at the origin of the sun, not the solar system's barycenter */
    r_ON_N = cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.r_BdyZero_N);
    r_SN_N = cArray2EigenVector3d(sunEphemerisInMsgBuffer.r_BdyZero_N);
    r_SO_O = dcm_ON * (r_SN_N - r_ON_N);  // small body to sun pos vector

    /* Compute the dcm from the body frame to the body's hill frame */
    double dcm_BN[3][3];
    MRP2C(navAttInMsgBuffer.sigma_BN, dcm_BN);
    Eigen::Matrix3d dcm_OB;
    dcm_OB = dcm_ON * (cArray2EigenMatrixXd(*dcm_BN, 3, 3));

    /* Compute x1, x2 from the input messages */
    double r_BO_O[3];
    double v_BO_O[3];
    rv2hill(asteroidEphemerisInMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N, navTransInMsgBuffer.r_BN_N, navTransInMsgBuffer.v_BN_N, r_BO_O, v_BO_O);
    x1 = cArray2EigenVector3d(r_BO_O);
    x2 = cArray2EigenVector3d(v_BO_O);

    /* Compute dx1 and dx2 */
    dx1 = x1 - x1_ref;
    dx2 = x2 - x2_ref;

    /* Now compute current f */
    f_curr =
            -F_ddot * o_hat_3_tilde * x1 - 2 * F_dot * o_hat_3_tilde * x2 -
            pow(F_dot, 2) * o_hat_3_tilde * o_hat_3_tilde * x1
            - mu_ast * x1 / pow(x1.norm(), 3)
            + mu_sun * (3 * (r_SO_O / r_SO_O.norm()) * (r_SO_O / r_SO_O.norm()).transpose() - I) * x1 /
              pow(r_SO_O.norm(), 3)
            + C_SRP * P_0 * (1 + rho) * (A_sc / M_sc) * pow(AU*1000.,2) * o_hat_1 / pow(r_SO_O.norm(), 2);

    /* Now compute reference f */
    f_ref =
            -F_ddot * o_hat_3_tilde * x1_ref - 2 * F_dot * o_hat_3_tilde * x2_ref -
            pow(F_dot, 2) * o_hat_3_tilde * o_hat_3_tilde * x1_ref
            - mu_ast * x1_ref / pow(x1_ref.norm(), 3)
            + mu_sun * (3 * (r_SO_O / r_SO_O.norm()) * (r_SO_O / r_SO_O.norm()).transpose() - I) * x1_ref /
              pow(r_SO_O.norm(), 3)
            + C_SRP * P_0 * (1 + rho) * (A_sc / M_sc) * pow(AU*1000.,2) * o_hat_1 / pow(r_SO_O.norm(), 2);

    /* Compute the thrust in the small body's hill frame */
    thrust_O = -(f_curr - f_ref) - K1 * dx1 - K2 * dx2;

    /* Compute the thrust in the s/c body frame */
    thrust_B = (dcm_OB.transpose()) * thrust_O;
}

/*! This is the main method that gets called every time the module is updated.  Provide an appropriate description.
    @return void
*/
void SmallBodyWaypointFeedback::UpdateState(uint64_t CurrentSimNanos)
{
    this->readMessages();
    this->computeControl(CurrentSimNanos);
    this->writeMessages(CurrentSimNanos);
    prevTime = CurrentSimNanos;
}

/*! This method reads the input messages each call of updateState
    @return void
*/
void SmallBodyWaypointFeedback::writeMessages(uint64_t CurrentSimNanos){
    /* Create the output message buffer */
    CmdForceBodyMsgPayload forceOutMsgBuffer;

    /* Zero the output message buffer */
    forceOutMsgBuffer = this->forceOutMsg.zeroMsgPayload;

    /* Assign the force */
    eigenVector3d2CArray(thrust_B, forceOutMsgBuffer.forceRequestBody);

    /* Write the message */
    this->forceOutMsg.write(&forceOutMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write the c-wrapped message */
    CmdForceBodyMsg_C_write(&forceOutMsgBuffer, &this->forceOutMsgC, this->moduleID, CurrentSimNanos);
}
