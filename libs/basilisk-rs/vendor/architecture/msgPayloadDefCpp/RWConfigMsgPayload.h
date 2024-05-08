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

#ifndef SIM_RW_CONFIG_MESSAGE_H
#define SIM_RW_CONFIG_MESSAGE_H

#include <Eigen/Dense>
#include "simulation/dynamics/reactionWheels/reactionWheelSupport.h"


/*! @brief Structure used to define the individual RW configuration data message*/
typedef struct
//@cond DOXYGEN_IGNORE
RWConfigMsgPayload
//@endcond
{
    Eigen::Vector3d rWB_B;      //!< [m], position vector of the RW relative to the spacecraft body frame
    Eigen::Vector3d gsHat_B;    //!< [-] spin axis unit vector in body frame
    Eigen::Vector3d w2Hat0_B;   //!< [-] initial torque axis unit vector in body frame
    Eigen::Vector3d w3Hat0_B;   //!< [-] initial gimbal axis unit vector in body frame
    double mass = 1.0;          //!< [kg], reaction wheel rotor mass
    double theta = 0.0;         //!< [rad], wheel angle
    double Omega = 0.0;         //!< [rad/s], wheel speed
    double Js = 1.0;            //!< [kg-m^2], spin axis gsHat rotor moment of inertia
    double Jt = 1.0;            //!< [kg-m^2], gtHat axis rotor moment of inertia
    double Jg = 1.0;            //!< [kg-m^2], ggHat axis rotor moment of inertia
    double U_s = 0.0;           //!< [kg-m], static imbalance
    double U_d = 0.0;           //!< [kg-m^2], dynamic imbalance
    double d = 0.0;             //!< [m], wheel center of mass offset from wheel frame origin
    double J13 = 0.0;           //!< [kg-m^2], x-z inertia of wheel about wheel center in wheel frame (imbalance)
    double u_current = 0.0;     //!< [N-m], current motor torque
    double u_max = -1;          //!< [N-m], Max torque, negative value turns off saturating the wheel
    double u_min = 0.0;         //!< [N-m], Min torque
    double fCoulomb = 0.0;      //!< [N-m], Coulomb friction torque magnitude
    double fStatic = 0.0;       //!< [N-m], Static friction torque magnitude
    double betaStatic = -1.0;    //!< Stribeck friction coefficient; For stribeck friction to be activiated, user must change this variable to a positive non-zero number.
    double cViscous = 0.0;      //!< [N-m-s/rad] Viscous fricion coefficient
    double omegaLimitCycle = 0.0001; //!< [rad/s], wheel speed that avoids limit cycle with friction
    double frictionTorque = 0.0; //!< [N-m] friction torque, this is a computed value, don't set it directly
    double omegaBefore = 0.0;   //!< [rad/s], wheel speed one time step before
    bool frictionStribeck = 0;  //!< [-] Boolenian to determine if stribeck friction model is used or not, 0 is non-stribeck, 1 is stribeck; Parameter is set internally.
    double Omega_max = -1.0;    //!< [rad/s], max wheel speed, negative values turn off wheel saturation
    double P_max = -1.0;        //!< [N-m/s], maximum wheel power, negative values turn off power limit
    RWModels RWModel = BalancedWheels;       //!< [-], Type of imbalance model to use
    Eigen::Vector3d aOmega;     //!< [-], parameter used in coupled jitter back substitution
    Eigen::Vector3d bOmega;     //!< [-], parameter used in coupled jitter back substitution
    double cOmega = 0.0;        //!< [-], parameter used in coupled jitter back substitution
    Eigen::Matrix3d IRWPntWc_B;         //!< RW inertia about point Wc in B frame components
    Eigen::Matrix3d IPrimeRWPntWc_B;    //!< RW inertia B-frame derivative
    Eigen::Vector3d rWcB_B;             //!< position of Wc relative to B in B-frame components
    Eigen::Matrix3d rTildeWcB_B;        //!< tilde matrix of r_WcB_B
    Eigen::Vector3d rPrimeWcB_B;        //!< B-frame derivative of r_WcB_B
    Eigen::Vector3d w2Hat_B;            //!< unit vector
    Eigen::Vector3d w3Hat_B;            //!< unit vector
    char label[10];             //!< [-], label name of the RW device being simulated
}RWConfigMsgPayload;



#endif
