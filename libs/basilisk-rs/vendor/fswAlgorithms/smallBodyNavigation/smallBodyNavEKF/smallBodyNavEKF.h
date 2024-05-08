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


#ifndef SMALLBODYNAVEKF_H
#define SMALLBODYNAVEKF_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/EphemerisMsg_C.h"
#include "cMsgCInterface/SmallBodyNavMsg_C.h"
#include "cMsgCInterface/CmdForceBodyMsg_C.h"
#include "architecture/msgPayloadDefCpp/THROutputMsgPayload.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"

/*! @brief This module estimates relative spacecraft position and velocity with respect to the body and attitude and attitude rate of the body wrt. the inertial frame
 */
class SmallBodyNavEKF: public SysModel {
public:
    SmallBodyNavEKF();
    ~SmallBodyNavEKF();

    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);  //!< Resets module
    void UpdateState(uint64_t CurrentSimNanos);  //!< Updates state
    void addThrusterToFilter(Message<THROutputMsgPayload> *tmpThrusterMsg);  //!< Adds thruster message

private:
    void readMessages(uint64_t CurrentSimNanos);  //!< Reads input messages
    void writeMessages(uint64_t CurrentSimNanos);  //!< Writes output messages
    void predict(uint64_t CurrentSimNanos);  //!< Prediction step of Kalman filter
    void aprioriState(uint64_t CurrentSimNanos);  //!< Computes the apriori state
    void aprioriCovar(uint64_t CurrentSimNanos);  //!< Computes the apriori covariance
    void checkMRPSwitching();  //!< Checks the MRPs for switching
    void computeDynamicsMatrix(Eigen::VectorXd x_hat);  //!< Computes the new dynamics matrix, A_k
    void measurementUpdate();  //!< Computes the measurement update for the EKF
    void computeEquationsOfMotion(Eigen::VectorXd x_hat, Eigen::MatrixXd Phi); //!< Computes the EOMs of the state and state transition matrix

public:
    ReadFunctor<NavTransMsgPayload> navTransInMsg;  //!< Translational nav input message
    ReadFunctor<NavAttMsgPayload> navAttInMsg;  //!< Attitude nav input message
    ReadFunctor<EphemerisMsgPayload> asteroidEphemerisInMsg;  //!< Small body ephemeris input message
    ReadFunctor<EphemerisMsgPayload> sunEphemerisInMsg;  //!< Sun ephemeris input message
    ReadFunctor<CmdForceBodyMsgPayload> cmdForceBodyInMsg;  //!< Command force body in message
    std::vector<ReadFunctor<THROutputMsgPayload>> thrusterInMsgs;  //!< thruster input msg vector

    Message<NavTransMsgPayload> navTransOutMsg;  //!< Translational nav output message
    Message<SmallBodyNavMsgPayload> smallBodyNavOutMsg;  //!< Small body nav output msg - states and covariances
    Message<EphemerisMsgPayload> asteroidEphemerisOutMsg;  //!< Small body ephemeris output message

    NavTransMsg_C navTransOutMsgC = {};  //!< C-wrapped Translational nav output message
    SmallBodyNavMsg_C smallBodyNavOutMsgC = {};  //!< C-wrapped Small body nav output msg - states and covariances
    EphemerisMsg_C asteroidEphemerisOutMsgC = {};  //!< C-wrapped Small body ephemeris output message

    BSKLogger bskLogger;  //!< -- BSK Logging

    double C_SRP;  //!< SRP scaling coefficient
    double P_0;  //!< SRP at 1 AU
    double rho;  //!< Surface reflectivity
    double A_sc;  //!< Surface area of the spacecraft
    double M_sc;  //!< Mass of the spacecraft
    double mu_ast;  //!< Gravitational constant of the asteroid
    Eigen::MatrixXd Q;  //!< Process Noise
    Eigen::MatrixXd R;  //!< Measurement Noise
    Eigen::VectorXd x_hat_k;  //!< Current state estimate
    Eigen::MatrixXd P_k;  //!< Current estimation error covariance


private:
    NavTransMsgPayload navTransInMsgBuffer;  //!< Message buffer for input translational nav message
    NavAttMsgPayload navAttInMsgBuffer;  //!< Message buffer for input attitude nav message
    EphemerisMsgPayload asteroidEphemerisInMsgBuffer;  //!< Message buffer for asteroid ephemeris
    EphemerisMsgPayload sunEphemerisInMsgBuffer;  //!< Message buffer for sun ephemeris
    std::vector<THROutputMsgPayload> thrusterInMsgBuffer; //!< Buffer for thruster force and torques
    CmdForceBodyMsgPayload cmdForceBodyInMsgBuffer; //!< Buffer for the commanded force input

    uint64_t prevTime;  //!< Previous time, ns
    uint64_t numStates;  //!< Number of states
    Eigen::Vector3d thrust_B;  //!< Thrust expressed in body-frame components
    Eigen::Vector3d cmdForce_B;  //!< External force expressed in body-frame components
    Eigen::VectorXd x_hat_dot_k;  //!< Rate of change of state estimate
    Eigen::VectorXd x_hat_k1_;  //!< Apriori state estimate for time k+1
    Eigen::VectorXd x_hat_k1;  //!< Update state estimate for time k+1
    Eigen::MatrixXd P_dot_k;  //!< Rate of change of estimation error covariance
    Eigen::MatrixXd P_k1_;  //!< Apriori estimation error covariance
    Eigen::MatrixXd P_k1;  //!< Updated estimation error covariance
    Eigen::MatrixXd A_k;  //!< State dynamics matrix
    Eigen::MatrixXd Phi_k;  //!< State transition matrix
    Eigen::MatrixXd Phi_dot_k;  //!< Rate of change of STM
    Eigen::MatrixXd L;  //!<
    Eigen::MatrixXd M;  //!<
    Eigen::MatrixXd H_k1;  //!< Jacobian of measurement model
    Eigen::MatrixXd I_full;  //!< numStates x numStates identity matrix
    Eigen::VectorXd k1;  //!< k1 constant for RK4 integration
    Eigen::VectorXd k2;  //!< k2 constant for RK4 integration
    Eigen::VectorXd k3;  //!< k3 constant for RK4 integration
    Eigen::VectorXd k4;  //!< k4 constant for RK4 integration
    Eigen::MatrixXd k1_phi;  //!< k1 STM constant for RK4 integration
    Eigen::MatrixXd k2_phi;  //!< k2 STM constant for RK4 integration
    Eigen::MatrixXd k3_phi;  //!< k3 STM constant for RK4 integration
    Eigen::MatrixXd k4_phi;  //!< k4 STM constant for RK4 integration
    bool newMeasurements;  //! Keeps track of whether or not new measurements have been written

    double mu_sun;  //!< Gravitational parameter of the sun
    Eigen::Matrix3d o_hat_3_tilde;  //!< Tilde matrix of the third asteroid orbit frame base vector
    Eigen::Vector3d o_hat_1;  //!< First asteroid orbit frame base vector
    Eigen::MatrixXd I;  //!< 3 x 3 identity matrix
    classicElements oe_ast;  //!< Orbital elements of the asteroid
    double F_dot;  //!< Time rate of change of true anomaly
    double F_ddot;  //!< Second time derivative of true anomaly
    Eigen::Matrix3d dcm_ON;  //!< DCM from the inertial frame to the small-body's hill frame
    Eigen::Vector3d r_SO_O;  //!< Vector from the small body's origin to the inertial frame origin in small-body hill frame components
};


#endif
