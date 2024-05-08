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


#ifndef SMALLBODYWAYPOINTFEEDBACK_H
#define SMALLBODYWAYPOINTFEEDBACK_H

#include "architecture/_GeneralModuleFiles/sys_model.h"
#include "architecture/msgPayloadDefC/NavTransMsgPayload.h"
#include "architecture/msgPayloadDefC/NavAttMsgPayload.h"
#include "architecture/msgPayloadDefC/EphemerisMsgPayload.h"
#include "architecture/msgPayloadDefC/CmdForceBodyMsgPayload.h"
#include "cMsgCInterface/CmdForceBodyMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/messaging/messaging.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/astroConstants.h"

/*! @brief This module is provides a Lyapunov feedback control law for waypoint to waypoint guidance and control about
 * a small body. The waypoints are defined in the Hill frame of the body.
 */
class SmallBodyWaypointFeedback: public SysModel {
public:
    SmallBodyWaypointFeedback();
    ~SmallBodyWaypointFeedback();

    void SelfInit();  //!< Self initialization for C-wrapped messages
    void Reset(uint64_t CurrentSimNanos);
    void UpdateState(uint64_t CurrentSimNanos);
    void readMessages();
    void computeControl(uint64_t CurrentSimNanos);
    void writeMessages(uint64_t CurrentSimNanos);

public:
    ReadFunctor<NavTransMsgPayload> navTransInMsg;  //!< translational navigation input message
    ReadFunctor<NavAttMsgPayload> navAttInMsg;  //!< attitude navigation input message
    ReadFunctor<EphemerisMsgPayload> asteroidEphemerisInMsg;  //!< asteroid ephemeris input message
    ReadFunctor<EphemerisMsgPayload> sunEphemerisInMsg;  //!< sun ephemeris input message

    Message<CmdForceBodyMsgPayload> forceOutMsg;  //!< force command output

    CmdForceBodyMsg_C forceOutMsgC = {};  //!< C-wrapped force output message

    BSKLogger bskLogger;              //!< -- BSK Logging

    double C_SRP;  //!< SRP scaling coefficient
    double P_0;  //!< SRP at 1 AU
    double rho;  //!< Surface reflectivity
    double A_sc;  //!< Surface area of the spacecraft
    double M_sc;  //!< Mass of the spacecraft
    Eigen::Matrix3d IHubPntC_B;  //!< sc inertia
    double mu_ast;  //!< Gravitational constant of the asteroid

    Eigen::Vector3d x1_ref;  //!< Desired Hill-frame position
    Eigen::Vector3d x2_ref;  //!< Desired Hill-frame velocity
    Eigen::Matrix3d K1;  //!< Position gain
    Eigen::Matrix3d K2;  //!< Velocity gain

private:
    NavTransMsgPayload navTransInMsgBuffer;  //!< local copy of message buffer
    NavAttMsgPayload navAttInMsgBuffer;  //!< local copy of message buffer
    EphemerisMsgPayload asteroidEphemerisInMsgBuffer;  //!< local copy of message buffer
    EphemerisMsgPayload sunEphemerisInMsgBuffer;  //!< local copy of message buffer

    uint64_t prevTime;  //!< Previous time, ns
    double mu_sun;  //!< Gravitational parameter of the sun
    Eigen::Matrix3d o_hat_3_tilde;  //!< Tilde matrix of the third asteroid orbit frame base vector
    Eigen::Vector3d o_hat_1;  //!< First asteroid orbit frame base vector
    Eigen::MatrixXd I;  //!< 3 x 3 identity matrix
    classicElements oe_ast;  //!< Orbital elements of the asteroid
    double F_dot;  //!< Time rate of change of true anomaly
    double F_ddot;  //!< Second time derivative of true anomaly
    Eigen::Vector3d r_BN_N;
    Eigen::Vector3d v_BN_N;
    Eigen::Vector3d v_ON_N;
    Eigen::Vector3d r_ON_N;
    Eigen::Vector3d r_SN_N;
    Eigen::Matrix3d dcm_ON;  //!< DCM from the inertial frame to the small-body's hill frame
    Eigen::Vector3d r_SO_O;  //!< Vector from the small body's origin to the inertial frame origin in small-body hill frame components
    Eigen::Vector3d f_curr;
    Eigen::Vector3d f_ref;
    Eigen::Vector3d x1;
    Eigen::Vector3d x2;
    Eigen::Vector3d dx1;
    Eigen::Vector3d dx2;
    Eigen::Vector3d thrust_O;
    Eigen::Vector3d thrust_B;

};


#endif
