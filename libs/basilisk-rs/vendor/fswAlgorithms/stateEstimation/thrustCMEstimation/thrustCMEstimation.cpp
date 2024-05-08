/*
 ISC License
 
 Copyright (c) 2023, Laboratory  for Atmospheric and Space Physics, University of Colorado at Boulder
 
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

#include "thrustCMEstimation.h"
#include <cmath>

ThrustCMEstimation::ThrustCMEstimation() = default;

ThrustCMEstimation::~ThrustCMEstimation() = default;

/*! Initialize C-wrapped output messages */
void ThrustCMEstimation::SelfInit(){
    VehicleConfigMsg_C_init(&this->vehConfigOutMsgC);
}

/*! Reset the flyby OD filter to an initial state and
 initializes the internal estimation matrices.
 @return void
 @param CurrentSimNanos The clock time at which the function was called (nanoseconds)
 */
void ThrustCMEstimation::Reset(uint64_t CurrentSimNanos)
{
    /*! - Check if the required message has not been connected */
    if (!this->thrusterConfigBInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR,  " thrusterConfigInMsg wasn't connected.");
    }
    if (!this->intFeedbackTorqueInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR,  " intFeedbackTorqueInMsg wasn't connected.");
    }
    if (!this->attGuidInMsg.isLinked()) {
        bskLogger.bskLog(BSK_ERROR,  " attGuidInMsg wasn't connected.");
    }
    if (this->vehConfigInMsg.isLinked()) {
        this->cmKnowledge = true;
    }
    else {
        this->cmKnowledge = false;
    }

    /*! Initialize estimated state and covariances based on user inputs */
    this->P.setZero();
    this->R.setZero();
    this->I.setZero();
    for (int i=0; i<3; ++i) {
        this->P(i,i) = this->P0[i];
        this->R(i,i) = this->R0[i];
        this->I(i,i) = 1;
    }
    this->r_CB_est = this->r_CB_B;
}

/*! Take the relative position measurements and outputs an estimate of the
 spacecraft states in the inertial frame.
 @return void
 @param CurrentSimNanos The clock time at which the function was called (nanoseconds)
 */
void ThrustCMEstimation::UpdateState(uint64_t CurrentSimNanos)
{
    /*! create output message buffers */
    VehicleConfigMsgPayload vehConfigOutBuffer = {};
    CMEstDataMsgPayload cmEstDataBuffer = {};

    /*! read and allocate the thrustConfigMsg */
    THRConfigMsgPayload thrConfigBuffer = this->thrusterConfigBInMsg();

    /*! compute thruster information in B-frame coordinates */
    Eigen::Vector3d r_TB_B = cArray2EigenVector3d(thrConfigBuffer.rThrust_B);
    Eigen::Vector3d T_B = thrConfigBuffer.maxThrust * cArray2EigenVector3d(thrConfigBuffer.tHatThrust_B);

    /*! compute error w.r.t. target attitude */
    AttGuidMsgPayload attGuidBuffer = this->attGuidInMsg();
    Eigen::Vector3d sigma_BR   = cArray2EigenVector3d(attGuidBuffer.sigma_BR);
    Eigen::Vector3d omega_BR_B = cArray2EigenVector3d(attGuidBuffer.omega_BR_B);
    double attError = pow(sigma_BR.squaredNorm() + omega_BR_B.squaredNorm(), 0.5);

    /*! read commanded torque msg */
    CmdTorqueBodyMsgPayload cmdTorqueBuffer = this->intFeedbackTorqueInMsg();
    Eigen::Vector3d L_B = -cArray2EigenVector3d(cmdTorqueBuffer.torqueRequestBody);

    Eigen::Vector3d y;       // measurement
    Eigen::Vector3d preFit;  // pre-fit residual
    Eigen::Vector3d postFit; // post-fit residual
    Eigen::Matrix3d H;       // linear model
    Eigen::Matrix3d K;       // kalman gain
    Eigen::Matrix3d S;

    /*! assign preFit and postFit residuals to NaN, rewrite them in case of measurement update */
    preFit  = {nan("1"), nan("1"), nan("1")};
    postFit = {nan("1"), nan("1"), nan("1")};

    if ((this->attGuidInMsg.isWritten()) && (attError < this->attitudeTol)) {

        /*! subtract torque about point B from measurement model */
        y = L_B - r_TB_B.cross(T_B);
        /*! H is the skew-symmetric matrix obtained from T_B */
        H = eigenTilde(T_B);
        /*! S is defined for convenience */
        S = H * this->P * H.transpose() + this->R;
        /*! Kalman gain */
        K = this->P * H.transpose() * S.inverse();

        /*! pre-fit residuals */
        preFit = y - H * this->r_CB_est;

        /*! measurement state update */
        this->r_CB_est = this->r_CB_est + K * preFit;
        /*! measurement covariance update  */
        this->P = (this->I - K * H) * this->P;

        /*! post-fit residuals */
        postFit = y - H * this->r_CB_est;
    }

    /*! compute state 1-sigma covariance */
    Eigen::Vector3d sigma;
    for (int i=0; i<3; ++i) {
        sigma[i] = pow(this->P(i,i), 0.5);
    }

    /*! write estimation data to msg buffer */
    cmEstDataBuffer.attError = attError;
    eigenVector3d2CArray(this->r_CB_est, cmEstDataBuffer.state);
    eigenVector3d2CArray(sigma, cmEstDataBuffer.covariance);
    eigenVector3d2CArray(preFit, cmEstDataBuffer.preFitRes);
    eigenVector3d2CArray(postFit, cmEstDataBuffer.postFitRes);
    Eigen::Vector3d r_CB_error;
    if (this->cmKnowledge) {
        VehicleConfigMsgPayload vehConfigBuffer = this->vehConfigInMsg();
        Eigen::Vector3d r_CB_B_true = cArray2EigenVector3d(vehConfigBuffer.CoM_B);
        r_CB_error = this->r_CB_est - r_CB_B_true;
        eigenVector3d2CArray(r_CB_error, cmEstDataBuffer.stateError);
    }
    else {
        r_CB_error = {nan("1"), nan("1"), nan("1")};
    }
    eigenVector3d2CArray(r_CB_error, cmEstDataBuffer.stateError);
    /*! write output msg */
    this->cmEstDataOutMsg.write(&cmEstDataBuffer, this->moduleID, CurrentSimNanos);

    /*! write CM location to vehicle config buffer msg */
    eigenVector3d2CArray(this->r_CB_est, vehConfigOutBuffer.CoM_B);
    /*! write output msg */
    this->vehConfigOutMsg.write(&vehConfigOutBuffer, this->moduleID, CurrentSimNanos);
    VehicleConfigMsg_C_write(&vehConfigOutBuffer, &this->vehConfigOutMsgC, this->moduleID, CurrentSimNanos);
}
