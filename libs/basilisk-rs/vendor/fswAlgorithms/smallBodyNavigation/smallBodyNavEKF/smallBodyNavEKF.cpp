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


#include "fswAlgorithms/smallBodyNavigation/smallBodyNavEKF/smallBodyNavEKF.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <cstring>
#include <math.h>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
SmallBodyNavEKF::SmallBodyNavEKF()
{
    this->numStates = 12;
    this->mu_sun = 1.327124e20;
    this->o_hat_3_tilde.setZero();
    this->o_hat_3_tilde(0, 1) = -1;
    this->o_hat_3_tilde(1, 0) = 1;
    this->o_hat_1 << 1, 0, 0;
    this->I.setIdentity(3,3);
    this->I_full.setIdentity(this->numStates, this->numStates);
    this->C_SRP = 1.0;
    this->P_0 = 4.56e-6;
    this->rho = 0.4;
    this->x_hat_dot_k.setZero(this->numStates);
    this->x_hat_k1_.setZero(this->numStates);
    this->x_hat_k1.setZero(this->numStates);
    this->x_hat_k.setZero(this->numStates);
    this->P_dot_k.setZero(this->numStates, this->numStates);
    this->P_k1_.setZero(this->numStates, this->numStates);
    this->P_k1.setZero(this->numStates, this->numStates);
    this->A_k.setIdentity(this->numStates, this->numStates);
    this->Phi_k.setIdentity(this->numStates, this->numStates);
    this->Phi_dot_k.setZero(this->numStates, this->numStates);
    this->L.setIdentity(this->numStates, this->numStates);
    this->M.setIdentity(this->numStates, this->numStates);
    this->H_k1.setIdentity(this->numStates, this->numStates);
    this->k1.setZero(this->numStates);
    this->k2.setZero(this->numStates);
    this->k3.setZero(this->numStates);
    this->k4.setZero(this->numStates);
    this->k1_phi.setZero(this->numStates, this->numStates);
    this->k2_phi.setZero(this->numStates, this->numStates);
    this->k3_phi.setZero(this->numStates, this->numStates);
    this->k4_phi.setZero(this->numStates, this->numStates);
    this->prevTime = 0;
    return;
}

/*! Module Destructor */
SmallBodyNavEKF::~SmallBodyNavEKF()
{
}

/*! Initialize C-wrapped output messages */
void SmallBodyNavEKF::SelfInit(){
    NavTransMsg_C_init(&this->navTransOutMsgC);
    SmallBodyNavMsg_C_init(&this->smallBodyNavOutMsgC);
    EphemerisMsg_C_init(&this->asteroidEphemerisOutMsgC);
}

/*! This method is used to reset the module and checks that required input messages are connect.
    @return void
*/
void SmallBodyNavEKF::Reset(uint64_t CurrentSimNanos)
{
    /* check that required input messages are connected */
    if (!this->navTransInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavEKF.navTransInMsg was not linked.");
    }
    if (!this->navAttInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavEKF.navAttInMsg was not linked.");
    }
    if (!this->asteroidEphemerisInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavEKF.asteroidEphemerisInMsg was not linked.");
    }
    if (!this->sunEphemerisInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavEKF.sunEphemerisInMsg was not linked.");
    }

}

/*! This method is used to add a thruster to the filter.
    @return void
*/
void SmallBodyNavEKF::addThrusterToFilter(Message<THROutputMsgPayload> *tmpThrusterMsg){
    this->thrusterInMsgs.push_back(tmpThrusterMsg->addSubscriber());
    return;
}

/*! This method is used to read the input messages.
    @param CurrentSimNanos
    @return void
*/
void SmallBodyNavEKF::readMessages(uint64_t CurrentSimNanos){
    /* Read in the input messages */
    if ((this->navTransInMsg.timeWritten() + this->navAttInMsg.timeWritten() + this->asteroidEphemerisInMsg.timeWritten() - 3*CurrentSimNanos) == 0){
        this->newMeasurements = true;
    } else {
        this->newMeasurements = false;
    }
    this->navTransInMsgBuffer = this->navTransInMsg();
    this->navAttInMsgBuffer = this->navAttInMsg();
    this->asteroidEphemerisInMsgBuffer = this->asteroidEphemerisInMsg();
    this->sunEphemerisInMsgBuffer = this->sunEphemerisInMsg();

    if (this->cmdForceBodyInMsg.isLinked()){
        this->cmdForceBodyInMsgBuffer= this->cmdForceBodyInMsg();
    } else{
        this->cmdForceBodyInMsgBuffer = this->cmdForceBodyInMsg.zeroMsgPayload;
    }


    /* Read the thruster messages */
    THROutputMsgPayload thrusterMsg;
    this->thrusterInMsgBuffer.clear();
    if (this->thrusterInMsgs.size() > 0){
        for (long unsigned int c = 0; c<this->thrusterInMsgs.size(); c++){
            thrusterMsg = this->thrusterInMsgs.at(c)();
            this->thrusterInMsgBuffer.push_back(thrusterMsg);
        }
    } else {
        bskLogger.bskLog(BSK_WARNING, "Small Body Nav EKF has no thruster messages to read.");
    }

}

/*! This method performs the KF prediction step
    @param CurrentSimNanos
    @return void
*/
void SmallBodyNavEKF::predict(uint64_t CurrentSimNanos){
    /* Get the orbital elements of the asteroid, we assume the uncertainty on the pos. and vel. of the body are low
     * enough to consider them known apriori */
    rv2elem(mu_sun, asteroidEphemerisInMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N, &oe_ast);

    /* Compute F_dot and F_ddot */
    F_dot = sqrt(mu_sun/(pow(oe_ast.a*(1-pow(oe_ast.e, 2)), 3)))*pow(1+(oe_ast.e)*cos(oe_ast.f), 2);
    F_ddot = -2*(oe_ast.e)*(sqrt(mu_sun/(pow(oe_ast.a*(1-pow(oe_ast.e, 2)), 3))))*sin(oe_ast.f)*(1+oe_ast.e*cos(oe_ast.f))*(F_dot);

    /* Compute the hill frame DCM of the small body */
    double dcm_ON_array[3][3];
    hillFrame(asteroidEphemerisInMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N, dcm_ON_array);
    dcm_ON = cArray2EigenMatrixXd(*dcm_ON_array, 3, 3).transpose();

    /* Compute the direction of the sun from the asteroid in the small body's hill frame, assumes heliocentric frame
     * centered at the origin of the sun, not the solar system's barycenter*/
    Eigen::Vector3d r_ON_N;  // inertial to small body pos. vector
    Eigen::Vector3d r_SN_N;  // inertial to sun pos. vector
    r_ON_N = cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.r_BdyZero_N);
    r_SN_N = cArray2EigenVector3d(sunEphemerisInMsgBuffer.r_BdyZero_N);
    r_SO_O = dcm_ON*(r_SN_N - r_ON_N);  // small body to sun pos vector

    /* Compute the total thrust and torque from the thrusters */
    thrust_B.setZero(); // Set to zero in case no thrusters are used
    for (long unsigned int c = 0; c < thrusterInMsgBuffer.size(); c++) {
        thrust_B += cArray2EigenVector3d(thrusterInMsgBuffer[c].thrustForce_B);
    }

    /* Add the commanded force */
    if (this->cmdForceBodyInMsg.isLinked()) {
        cmdForce_B = cArray2EigenVector3d(cmdForceBodyInMsgBuffer.forceRequestBody);
    }

    /* Compute aprior state estimate */
    aprioriState(CurrentSimNanos);

    /* Compute apriori covariance */
    aprioriCovar(CurrentSimNanos);
}

/*! This method computes the apriori state estimate using RK4 integration
    @param CurrentSimNanos
    @return void
*/
void SmallBodyNavEKF::aprioriState(uint64_t CurrentSimNanos){
    /* First RK4 step */
    computeEquationsOfMotion(x_hat_k, Phi_k);
    k1 = (CurrentSimNanos-prevTime)*NANO2SEC*x_hat_dot_k;
    k1_phi = (CurrentSimNanos-prevTime)*NANO2SEC*Phi_dot_k;

    /* Second RK4 step */
    computeEquationsOfMotion(x_hat_k + k1/2, Phi_k + k1_phi/2);
    k2 = (CurrentSimNanos-prevTime)*NANO2SEC*x_hat_dot_k;
    k2_phi = (CurrentSimNanos-prevTime)*NANO2SEC*Phi_dot_k;

    /* Third RK4 step */
    computeEquationsOfMotion(x_hat_k + k2/2, Phi_k + k2_phi/2);
    k3 = (CurrentSimNanos-prevTime)*NANO2SEC*x_hat_dot_k;
    k3_phi = (CurrentSimNanos-prevTime)*NANO2SEC*Phi_dot_k;

    /* Fourth RK4 step */
    computeEquationsOfMotion(x_hat_k + k3, Phi_k + k3_phi);
    k4 = (CurrentSimNanos-prevTime)*NANO2SEC*x_hat_dot_k;
    k4_phi = (CurrentSimNanos-prevTime)*NANO2SEC*Phi_dot_k;

    /* Perform the RK4 integration on the dynamics and STM */
    x_hat_k1_ = x_hat_k + (k1 + 2*k2 + 2*k3 + k4)/6;
    Phi_k = Phi_k + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)/6;
}

/*! This method calculates the EOMs of the state vector and state transition matrix
    @param x_hat
    @param Phi
    @return void
*/
void SmallBodyNavEKF::computeEquationsOfMotion(Eigen::VectorXd x_hat, Eigen::MatrixXd Phi){
    /* Create temporary state vectors for readability */
    Eigen::Vector3d x_1;
    Eigen::Vector3d x_2;
    Eigen::Vector3d x_3;
    Eigen::Vector3d x_4;

    x_1 << x_hat.segment(0,3);
    x_2 << x_hat.segment(3,3);
    x_3 << x_hat.segment(6,3);
    x_4 << x_hat.segment(9,3);

    /* x1_dot */
    x_hat_dot_k.segment(0,3) = x_2;

    /* x2_dot */
    /* First compute dcm_OB, DCM from sc body-frame to orbit frame*/
    double dcm_BN_meas[3][3];
    MRP2C(this->navAttInMsgBuffer.sigma_BN, dcm_BN_meas);
    Eigen::Matrix3d dcm_OB;
    dcm_OB = dcm_ON*(cArray2EigenMatrixXd(*dcm_BN_meas, 3, 3));
    /* Now compute x2_dot */
    x_hat_dot_k.segment(3,3) =
            -F_ddot*o_hat_3_tilde*x_1 - 2*F_dot*o_hat_3_tilde*x_2 -pow(F_dot,2)*o_hat_3_tilde*o_hat_3_tilde*x_1
            - mu_ast*x_1/pow(x_1.norm(), 3)
            + mu_sun*(3*(r_SO_O/r_SO_O.norm())*(r_SO_O/r_SO_O.norm()).transpose()-I)*x_1/pow(r_SO_O.norm(), 3)
            + C_SRP*P_0*(1+rho)*(A_sc/M_sc)*o_hat_1/pow(r_SO_O.norm(), 2)
            + dcm_OB*thrust_B/M_sc
            + dcm_OB*cmdForce_B/M_sc;

    /* x3_dot */
    x_hat_dot_k.segment(6,3) = 0.25*((1-pow(x_3.norm(),2))*I + 2*eigenTilde(x_3) + 2*x_3*x_3.transpose())*x_4;

    /* x4_dot */
    x_hat_dot_k.segment(9,3) << 0, 0, 0;

    /* Re-compute the dynamics matrix and compute Phi_dot */
    computeDynamicsMatrix(x_hat);
    Phi_dot_k = A_k*Phi;
}

/*! This method compute the apriori estimation error covariance through euler integration
    @param CurrentSimNanos
    @return void
*/
void SmallBodyNavEKF::aprioriCovar(uint64_t CurrentSimNanos){
    /* Compute the apriori covariance */
    P_k1_ = Phi_k*P_k*Phi_k.transpose() + L*Q*L.transpose();
}

/*! This method checks the propagated MRP states to see if they exceed a norm of 1. If they do, the appropriate
    states are transferred to the shadow set and the covariance is updated.
    @return void
 */
void SmallBodyNavEKF::checkMRPSwitching(){
    /* Create temporary values for sigma_AN */
    Eigen::Vector3d sigma_AN;
    sigma_AN << x_hat_k1_.segment(6,3);

    /* Create a shadow covariance matrix */
    Eigen::MatrixXd P_k1_s;
    P_k1_s.setZero(this->numStates, this->numStates);

    /* Initialize Lambda, set it to zero, set diagonal 3x3 state blocks to identity */
    Eigen::MatrixXd Lambda;
    Lambda.setZero(this->numStates, this->numStates);
    Lambda.block(0, 0, 3, 3).setIdentity();
    Lambda.block(3, 3, 3, 3).setIdentity();
    Lambda.block(6, 6, 3, 3).setIdentity();
    Lambda.block(9, 9, 3, 3).setIdentity();

    /* Check the attitude of the small body */
    if (sigma_AN.norm() > 1.0){
        /* Switch MRPs */
        x_hat_k1_.segment(6, 3) = -sigma_AN/pow(sigma_AN.norm(), 2);
        /* Populate lambda block */
        Lambda.block(6, 6, 3, 3) = 2*sigma_AN*sigma_AN.transpose()/pow(sigma_AN.norm(), 4) - I/pow(sigma_AN.norm(), 2);
    }

    /* Compute the new apriori covariance */
    P_k1_s = Lambda*P_k1_*Lambda.transpose();
    P_k1_ = P_k1_s;
}


/*! This method performs the KF measurement update step
    @return void
*/
void SmallBodyNavEKF::measurementUpdate(){
    /* Compute Kalman gain */
    Eigen::MatrixXd K_k1;
    K_k1.setZero(this->numStates, this->numStates);
    K_k1 = P_k1_*H_k1.transpose() * (H_k1*P_k1_*H_k1.transpose() + M*R*M.transpose()).inverse();

    /* Grab the measurements from the input messages */
    /* Subtract the asteroid position from the spacecraft position and rotate it into the small body's hill frame*/
    Eigen::VectorXd y_k1;
    y_k1.setZero(this->numStates);
    y_k1.segment(0, 3) = dcm_ON*(cArray2EigenVector3d(navTransInMsgBuffer.r_BN_N)
            - cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.r_BdyZero_N));

    /* Perform a similar operation for the relative velocity */
    y_k1.segment(3, 3) = dcm_ON*(cArray2EigenVector3d(navTransInMsgBuffer.v_BN_N)
            - cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.v_BdyZero_N));

    /* Small body attitude from the ephemeris msg */
    y_k1.segment(6, 3) =  cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.sigma_BN);

    /* Check if the shadow set measurement must be considered, i.e. |sigma| > 1/3 */
    if (y_k1.segment(6, 3).norm() > 1.0/3.0) {
        /* Create a temporary shadow-set MRP representation */
        Eigen::Vector3d sigma_AN_s = -y_k1.segment(6, 3)/pow(y_k1.segment(6, 3).norm(), 2);
        /* Check to see if the shadow set gives a smaller residual */
        if ((sigma_AN_s - x_hat_k1_.segment(6, 3)).norm() < (y_k1.segment(6, 3) - x_hat_k1_.segment(6, 3)).norm()){
            y_k1.segment(6, 3) = sigma_AN_s;
        }
    }

    /* Small body attitude rate from the ephemeris msg */
    y_k1.segment(9, 3) = cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.omega_BN_B);

    /* Update the state estimate */
    x_hat_k1 = x_hat_k1_ + K_k1*(y_k1 - x_hat_k1_);

    /* Update the covariance */
    P_k1 = (I_full - K_k1*H_k1)*P_k1_*(I_full - K_k1*H_k1).transpose() + K_k1*M*R*M.transpose()*K_k1.transpose();

    /* Assign the state estimate and covariance to k for the next iteration */
    x_hat_k = x_hat_k1;
    P_k = P_k1;
}

/*! This method computes the state dynamics matrix, A, for the next iteration
    @return void
*/
void SmallBodyNavEKF::computeDynamicsMatrix(Eigen::VectorXd x_hat){
    /* Create temporary state vectors for readability */
    Eigen::Vector3d x_1;
    Eigen::Vector3d x_2;
    Eigen::Vector3d x_3;
    Eigen::Vector3d x_4;

    x_1 << x_hat.segment(0,3);
    x_2 << x_hat.segment(3,3);
    x_3 << x_hat.segment(6,3);
    x_4 << x_hat.segment(9,3);

    /* First set the matrix to zero (many indices are zero) */
    A_k.setZero(this->numStates, this->numStates);

    /* x_1 partial */
    A_k.block(0, 3, 3, 3).setIdentity();

    /* x_2 partial */
    A_k.block(3, 0, 3, 3) =
            - F_ddot*o_hat_3_tilde
            - pow(F_dot, 2)*o_hat_3_tilde*o_hat_3_tilde
            - mu_ast/pow(x_1.norm(), 3)*I
            + 3*mu_ast*x_1*x_1.transpose()/pow(x_1.norm(), 5)
            + mu_sun*(3*(r_SO_O*r_SO_O.transpose())/pow(r_SO_O.norm(), 2) - I)/pow(r_SO_O.norm(), 3);

    A_k.block(3, 3, 3, 3) = -2*F_dot*o_hat_3_tilde;

    /* x_3 partial */
    A_k.block(6, 6, 3, 3) = 0.5*(x_3*x_4.transpose() - x_4*x_3.transpose() - eigenTilde(x_4) + (x_4.transpose()*x_3)*I);
    A_k.block(6, 9, 3, 3) = 0.25*((1-pow(x_3.norm(), 2))*I + 2*eigenTilde(x_3) + 3*x_3*x_3.transpose());
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallBodyNavEKF::UpdateState(uint64_t CurrentSimNanos)
{
    this->readMessages(CurrentSimNanos);
    this->predict(CurrentSimNanos);
    this->checkMRPSwitching();
    if (this->newMeasurements){
        /* Run the measurement update */
        this->measurementUpdate();
    }
    else{
        /* Assign the apriori state estimate and covariance to k for the next iteration */
        x_hat_k = x_hat_k1_;
    }
    this->writeMessages(CurrentSimNanos);
    prevTime = CurrentSimNanos;
}

/*! This method writes the output messages
    @return void
*/
void SmallBodyNavEKF::writeMessages(uint64_t CurrentSimNanos){
    /* Create output msg buffers */
    NavTransMsgPayload navTransOutMsgBuffer;
    SmallBodyNavMsgPayload smallBodyNavOutMsgBuffer;
    EphemerisMsgPayload asteroidEphemerisOutMsgBuffer;

    /* Zero the output message buffers before assigning values */
    navTransOutMsgBuffer = this->navTransOutMsg.zeroMsgPayload;
    smallBodyNavOutMsgBuffer = this->smallBodyNavOutMsg.zeroMsgPayload;
    asteroidEphemerisOutMsgBuffer = this->asteroidEphemerisOutMsg.zeroMsgPayload;

    /* Assign values to the nav trans output message */
    navTransOutMsgBuffer.timeTag = navTransInMsgBuffer.timeTag;
    eigenMatrixXd2CArray(cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.r_BdyZero_N) + dcm_ON.transpose()*x_hat_k.segment(0,3), navTransOutMsgBuffer.r_BN_N);
    eigenMatrixXd2CArray(cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.v_BdyZero_N) + dcm_ON.transpose()*x_hat_k.segment(3,3), navTransOutMsgBuffer.v_BN_N);
    v3Copy(navTransOutMsgBuffer.vehAccumDV, navTransInMsgBuffer.vehAccumDV);  // Not an estimated parameter, pass through

    /* Assign values to the asteroid ephemeris output message */
    v3Copy(asteroidEphemerisOutMsgBuffer.r_BdyZero_N, asteroidEphemerisInMsgBuffer.r_BdyZero_N);  // Not an estimated parameter
    v3Copy(asteroidEphemerisOutMsgBuffer.v_BdyZero_N, asteroidEphemerisInMsgBuffer.v_BdyZero_N);  // Not an estimated parameter
    eigenMatrixXd2CArray(x_hat_k.segment(6,3), asteroidEphemerisOutMsgBuffer.sigma_BN);
    eigenMatrixXd2CArray(x_hat_k.segment(9,3), asteroidEphemerisOutMsgBuffer.omega_BN_B);
    asteroidEphemerisOutMsgBuffer.timeTag = asteroidEphemerisInMsgBuffer.timeTag;

    /* Assign values to the small body navigation output message */
    eigenMatrixXd2CArray(x_hat_k, smallBodyNavOutMsgBuffer.state);
    if (this->newMeasurements) {
        eigenMatrixXd2CArray(P_k, *smallBodyNavOutMsgBuffer.covar);
    } else {
        eigenMatrixXd2CArray(P_k1_, *smallBodyNavOutMsgBuffer.covar);
    }

    /* Write to the C++-wrapped output messages */
    this->navTransOutMsg.write(&navTransOutMsgBuffer, this->moduleID, CurrentSimNanos);
    this->smallBodyNavOutMsg.write(&smallBodyNavOutMsgBuffer, this->moduleID, CurrentSimNanos);
    this->asteroidEphemerisOutMsg.write(&asteroidEphemerisOutMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write to the C-wrapped output messages */
    NavTransMsg_C_write(&navTransOutMsgBuffer, &this->navTransOutMsgC, this->moduleID, CurrentSimNanos);
    SmallBodyNavMsg_C_write(&smallBodyNavOutMsgBuffer, &this->smallBodyNavOutMsgC, this->moduleID, CurrentSimNanos);
    EphemerisMsg_C_write(&asteroidEphemerisOutMsgBuffer, &this->asteroidEphemerisOutMsgC, this->moduleID, CurrentSimNanos);
}

