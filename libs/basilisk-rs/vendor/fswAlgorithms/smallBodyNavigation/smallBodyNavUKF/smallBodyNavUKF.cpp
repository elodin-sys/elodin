/*
 ISC License

 Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#include "fswAlgorithms/smallBodyNavigation/smallBodyNavUKF/smallBodyNavUKF.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include <iostream>
#include <cstring>
#include <math.h>

/*! This is the constructor for the module class.  It sets default variable
    values and initializes the various parts of the model */
SmallBodyNavUKF::SmallBodyNavUKF()
{
    this->numStates = 9;
    this->numMeas = 3;
    this->numSigmas = 2*this->numStates + 1;
    this->x_hat_k1_.setZero(this->numStates);
    this->P_k1_.setZero(this->numStates, this->numStates);
    this->x_hat_k1.setZero(this->numStates);
    this->P_k1.setZero(this->numStates, this->numStates);
    this->X_sigma_k1_.setZero(this->numStates, this->numSigmas);
    this->wm_sigma.setZero(this->numSigmas);
    this->wc_sigma.setZero(this->numSigmas);
    this->y_hat_k1_.setZero(this->numMeas);
    this->R_k1_.setZero(this->numMeas, this->numMeas);
    this->Y_sigma_k1_.setZero(this->numMeas, this->numSigmas);
    this->H.setZero(this->numStates, this->numMeas);
    this->K.setZero(this->numStates, this->numMeas);
    this->dcm_AN.setIdentity(3,3);
    this->omega_AN_A.setZero(3);
    this->alpha = 0;
    this->beta = 2;
    this->kappa = 1e-3;
    this->prevTime = 0;
    return;
}

/*! Module Destructor */
SmallBodyNavUKF::~SmallBodyNavUKF()
{
}

/*! Initialize C-wrapped output messages */
void SmallBodyNavUKF::SelfInit(){
    SmallBodyNavUKFMsg_C_init(&this->smallBodyNavUKFOutMsgC);
}

/*! This method is used to reset the module, check that required input messages are connect and compute weigths.
    @return void
*/
void SmallBodyNavUKF::Reset(uint64_t CurrentSimNanos)
{
    /* check that required input messages are connected */
    if (!this->navTransInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavUKF.navTransInMsg was not linked.");
    }
    if (!this->asteroidEphemerisInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR, "SmallBodyNavUKF.asteroidEphemerisInMsg was not linked.");
    }
    
    /* compute UT weights to be used in the UT */
    this->wm_sigma(0) = this->kappa / (this->kappa + this->numStates);
    this->wc_sigma(0) = this->wm_sigma(0) + 1 - pow(this->alpha,2) + this->beta;
    for (int i = 0; i < this->numStates; i++) {
        /* Assign weigths */
        this->wm_sigma(i+1) = 1 / (2*(this->numStates + this->kappa));
        this->wm_sigma(numStates+i+1) = this->wm_sigma(i+1);
        this->wc_sigma(i+1) = this->wm_sigma(i+1);
        this->wc_sigma(numStates+i+1) = this->wm_sigma(i+1);
    }
}

/*! This method is used to read the input messages.
    @return void
*/
void SmallBodyNavUKF::readMessages(){
    /* Read in the input messages */
    this->navTransInMsgBuffer = this->navTransInMsg();
    this->asteroidEphemerisInMsgBuffer = this->asteroidEphemerisInMsg();
}

/*! This method does the UT to the initial distribution to compute the a-priori state
    @param CurrentSimNanos
    @return void
*/
void SmallBodyNavUKF::processUT(uint64_t CurrentSimNanos){
    /* Read angular velocity of the small body fixed frame */
    this->omega_AN_A = cArray2EigenVector3d(this->asteroidEphemerisInMsgBuffer.omega_BN_B);
    
    /* Declare matrix to store sigma points spread */
    Eigen::MatrixXd X_sigma_k;
    Eigen::MatrixXd X_sigma_dot_k;
    
    /* Set sigma points related matrices and vectors to zero */
    X_sigma_k.setZero(this->numStates, this->numSigmas);
    X_sigma_dot_k.setZero(this->numStates, this->numSigmas);
    
    /* Compute square root matrix of covariance */
    Eigen::MatrixXd Psqrt_k;
    Psqrt_k = this->P_k.llt().matrixL();
    
    /* Assign mean to central sigma point */
    X_sigma_k.col(0) = this->x_hat_k;
    
    /* Loop to generate remaining sigma points */
    for (int i = 0; i < this->numStates; i++) {
        /* Generate sigma points */
        X_sigma_k.col(i+1) = this->x_hat_k
            - sqrt(this->numStates + this->kappa) * Psqrt_k.col(i);
        X_sigma_k.col(numStates+i+1) = x_hat_k
            + sqrt(this->numStates + this->kappa) * Psqrt_k.col(i);
    }
    
    /* Loop to propagate sigma points and compute mean */
    Eigen::VectorXd x_sigma_k;
    Eigen::VectorXd x_sigma_dot_k;
    Eigen::Vector3d r_sigma_k;
    Eigen::Vector3d v_sigma_k;
    Eigen::Vector3d a_sigma_k;
    x_sigma_k.setZero(this->numStates);
    x_sigma_dot_k.setZero(this->numStates);
    this->x_hat_k1_.setZero(this->numStates);
    for (int i = 0; i < this->numSigmas; i++) {
        /* Extract sigma point */
        x_sigma_k = X_sigma_k.col(i);
        
        /* Compute dynamics derivative */
        r_sigma_k << x_sigma_k.segment(0,3);
        v_sigma_k << x_sigma_k.segment(3,3);
        a_sigma_k << x_sigma_k.segment(6,3);
        x_sigma_dot_k.segment(0,3) = v_sigma_k;
        x_sigma_dot_k.segment(3,3) = - 2*this->omega_AN_A.cross(v_sigma_k)
                                     - this->omega_AN_A.cross(this->omega_AN_A.cross(r_sigma_k))
                                     - this->mu_ast*r_sigma_k/pow(r_sigma_k.norm(), 3)
                                     + a_sigma_k;
        
        /* Use Euler integration to propagate */
        this->X_sigma_k1_.col(i) = x_sigma_k + x_sigma_dot_k*(CurrentSimNanos-prevTime)*NANO2SEC;
        
        /* Compute average */
        this->x_hat_k1_ = this->x_hat_k1_ + this->wm_sigma(i)*this->X_sigma_k1_.col(i);
    }
    
    /* Loop to compute covariance */
    Eigen::VectorXd x_sigma_dev_k1_;
    x_sigma_dev_k1_.setZero(this->numStates);
    this->P_k1_.setZero(this->numStates, this->numStates);
    for (int i = 0; i < numSigmas; i++) {
        /* Compute deviation of sigma from the mean */
        x_sigma_dev_k1_ = this->X_sigma_k1_.col(i) - this->x_hat_k1_;
        
        /* Add the deviation to the covariance */
        this->P_k1_ = this->P_k1_ + this->wc_sigma(i)*x_sigma_dev_k1_*x_sigma_dev_k1_.transpose();
    }
    
    /* Add process noise covariance */
    this->P_k1_ = this->P_k1_ + this->P_proc;
}

/*! This method does the UT to the a-priori state to compute the a-priori measurements
    @return void
*/
void SmallBodyNavUKF::measurementUT(){
    /* Compute square root matrix of covariance */
    Eigen::MatrixXd Psqrt_k1_;
    Psqrt_k1_ = P_k1_.llt().matrixL();
    
    /* Assign mean to central sigma point */
    this->X_sigma_k1_.col(0) = this->x_hat_k1_;
    
    /* Loop to generate remaining sigma points */
    for (int i = 0; i < this->numStates; i++) {
        /* Generate sigma points */
        this->X_sigma_k1_.col(i+1) = this->x_hat_k1_
            - sqrt(this->numStates + this->kappa) * Psqrt_k1_.col(i);
        this->X_sigma_k1_.col(this->numStates+i+1) = this->x_hat_k1_
            + sqrt(this->numStates + this->kappa) * Psqrt_k1_.col(i);
    }
    
    /* Loop to propagate sigma points and compute mean */
    Eigen::VectorXd x_sigma_k1_;
    x_sigma_k1_.setZero(this->numStates);
    this->y_hat_k1_.setZero(this->numMeas);
    for (int i = 0; i < this->numSigmas; i++) {
        /* Extract sigma point */
        x_sigma_k1_ = this->X_sigma_k1_.col(i);
        
        /* Assign correlation between state and measurement */
        this->Y_sigma_k1_.col(i) = x_sigma_k1_.segment(0,3);
        
        /* Compute average */
        this->y_hat_k1_ = this->y_hat_k1_ + this->wm_sigma(i)*this->Y_sigma_k1_.col(i);
    }
    
    /* Loop to compute measurements covariance and cross-correlation */
    Eigen::VectorXd x_sigma_dev_k1_;
    Eigen::VectorXd y_sigma_dev_k1_;
    x_sigma_dev_k1_.setZero(this->numStates);
    y_sigma_dev_k1_.setZero(this->numStates);
    this->R_k1_.setZero(this->numMeas, this->numMeas);
    this->H.setZero(this->numStates, this->numMeas);
    for (int i = 0; i < this->numSigmas; i++) {
        /* Compute deviation of measurement sigma from the mean */
        x_sigma_dev_k1_ = this->X_sigma_k1_.col(i) - this->x_hat_k1_;
        y_sigma_dev_k1_ = this->Y_sigma_k1_.col(i) - this->y_hat_k1_;
        
        /* Add the deviation to the measurement and cross-correlation covariances*/
        this->R_k1_ = this->R_k1_ + this->wc_sigma(i)*y_sigma_dev_k1_*y_sigma_dev_k1_.transpose();
        this->H = this->H + this->wc_sigma(i)*x_sigma_dev_k1_*y_sigma_dev_k1_.transpose();
    }
    
    /* Extract dcm of the small body, it transforms from inertial to small body fixed frame */
    double dcm_AN_array[3][3];
    MRP2C(asteroidEphemerisInMsgBuffer.sigma_BN, dcm_AN_array);
    this->dcm_AN = cArray2EigenMatrix3d(*dcm_AN_array);
    
    /* Add process noise covariance */
    this->R_k1_ = this->R_k1_ + this->dcm_AN * this->R_meas * this->dcm_AN.transpose();
}

/*! This method collects the measurements and updates the estimation
    @return void
*/
void SmallBodyNavUKF::kalmanUpdate(){
    /* Read attitude MRP of the small body fixed frame w.r.t. inertial */
    Eigen::Vector3d sigma_AN;
    sigma_AN = cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.sigma_BN);
    
    /* Subtract the asteroid position from the spacecraft position */
    Eigen::VectorXd y_k1;
    y_k1.setZero(this->numMeas);
    y_k1.segment(0, 3) = this->dcm_AN*(cArray2EigenVector3d(navTransInMsgBuffer.r_BN_N) -  cArray2EigenVector3d(asteroidEphemerisInMsgBuffer.r_BdyZero_N));
    
    /* Compute Kalman gain */
    this->K = this->H*this->R_k1_.inverse();
    
    /* Compute the Kalman innovation */
    Eigen::VectorXd w_k1;
    w_k1.setZero(this->numStates);
    w_k1 = this->K * (y_k1 - this->y_hat_k1_);
    
    /* Update state estimation and covariance */
    this->x_hat_k1 = this->x_hat_k1_ + w_k1;
    this->P_k1 = this->P_k1_ - this->K * this->R_k1_ * this->K.transpose();
    
    /* Assign the state estimate and covariance to k for the next iteration */
    this->x_hat_k = this->x_hat_k1;
    this->P_k = this->P_k1;
}

/*! This method writes the output messages
    @return void
*/
void SmallBodyNavUKF::writeMessages(uint64_t CurrentSimNanos){
    /* Create output msg buffers */
    SmallBodyNavUKFMsgPayload smallBodyNavUKFOutMsgBuffer;
    
    /* Zero the output message buffers before assigning values */
    smallBodyNavUKFOutMsgBuffer = this->smallBodyNavUKFOutMsg.zeroMsgPayload;
    
    /* Assign values to the small body navigation output message */
    eigenMatrixXd2CArray(this->x_hat_k1, smallBodyNavUKFOutMsgBuffer.state);
    eigenMatrixXd2CArray(this->P_k1, *smallBodyNavUKFOutMsgBuffer.covar);
    
    /* Write to the C++-wrapped output messages */
    this->smallBodyNavUKFOutMsg.write(&smallBodyNavUKFOutMsgBuffer, this->moduleID, CurrentSimNanos);

    /* Write to the C-wrapped output messages */
    SmallBodyNavUKFMsg_C_write(&smallBodyNavUKFOutMsgBuffer, &this->smallBodyNavUKFOutMsgC, this->moduleID, CurrentSimNanos);
}

/*! This is the main method that gets called every time the module is updated.
    @return void
*/
void SmallBodyNavUKF::UpdateState(uint64_t CurrentSimNanos)
{
    this->readMessages();
    this->processUT(CurrentSimNanos);
    this->measurementUT();
    this->kalmanUpdate();
    this->writeMessages(CurrentSimNanos);
    this->prevTime = CurrentSimNanos;
}
