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
/*
    FSW Electrostatic Tractor Control
 
 */

/* modify the path to reflect the new module names */
#include "etSphericalControl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 Pull in support files from other modules.  Be sure to use the absolute path relative to Basilisk directory.
 */
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/astroConstants.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/rigidBodyKinematics.h"



/*!
    This method initializes the output messages for this module.
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The module identifier
 */
void SelfInit_etSphericalControl(etSphericalControlConfig *configData, int64_t moduleID)
{
    CmdForceInertialMsg_C_init(&configData->forceInertialOutMsg);
    CmdForceBodyMsg_C_init(&configData->forceBodyOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime [ns] time the method is called
 @param moduleID The module identifier
*/
void Reset_etSphericalControl(etSphericalControlConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!NavTransMsg_C_isLinked(&configData->servicerTransInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.servicerTransInMsg wasn't connected.");
    }
    if (!NavTransMsg_C_isLinked(&configData->debrisTransInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.debrisTransInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->servicerAttInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.servicerAttInMsg wasn't connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->servicerVehicleConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.servicerVehicleConfigInMsg wasn't connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->debrisVehicleConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.debrisVehicleConfigInMsg wasn't connected.");
    }
    if (!CmdForceInertialMsg_C_isLinked(&configData->eForceInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: etSphericalControl.eForceInMsg wasn't connected.");
    }
    // check if input parameters are valid
    
    // L_r must be a positive value
    if (configData->L_r <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in etSphericalControl: L_r must be set to a positive value.");
    }
    
    // m_T and m_D must be positive and non-zero
    if (VehicleConfigMsg_C_read(&configData->servicerVehicleConfigInMsg).massSC <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in etSphericalControl: servicer mass must be set to a positive value.");
    }
    if (VehicleConfigMsg_C_read(&configData->debrisVehicleConfigInMsg).massSC <= 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in etSphericalControl: debris mass must be set to a positive value.");
    }
    // [K] and [P] must be positive definite matrices
    double EigenValuesK[3];
    double EigenValuesP[3];
    m33EigenValues(RECAST3X3 configData->K, EigenValuesK);
    m33EigenValues(RECAST3X3 configData->P, EigenValuesP);
    v3Scale(-1., EigenValuesK, EigenValuesK);
    v3Scale(-1., EigenValuesP, EigenValuesP);
    if (vMax(EigenValuesK, 3) > 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in etSphericalControl: K must be a positive definite 3 by 3 matrix.");
    }
    if (vMax(EigenValuesP, 3) > 0.0) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error in etSphericalControl: P must be a positive definite 3 by 3 matrix.");
    }

    return;
}

/*! Add a description of what this main Update() routine does for this module
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
*/
void Update_etSphericalControl(etSphericalControlConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // in
    NavTransMsgPayload servicerTransInMsgBuffer;
    NavTransMsgPayload debrisTransInMsgBuffer;
    NavAttMsgPayload servicerAttInMsgBuffer;
    VehicleConfigMsgPayload servicerVehicleConfigInMsgBuffer;
    VehicleConfigMsgPayload debrisVehicleConfigInMsgBuffer;
    CmdForceInertialMsgPayload eForceInMsgBuffer;
    // out
    CmdForceInertialMsgPayload forceInertialOutMsgBuffer;
    CmdForceBodyMsgPayload forceBodyOutMsgBuffer;
    
    // - always zero the output buffer first
    forceInertialOutMsgBuffer = CmdForceInertialMsg_C_zeroMsgPayload();
    forceBodyOutMsgBuffer = CmdForceBodyMsg_C_zeroMsgPayload();

    // - Read the input messages
    servicerTransInMsgBuffer = NavTransMsg_C_read(&configData->servicerTransInMsg);
    servicerAttInMsgBuffer = NavAttMsg_C_read(&configData->servicerAttInMsg);
    debrisTransInMsgBuffer = NavTransMsg_C_read(&configData->debrisTransInMsg);
    servicerVehicleConfigInMsgBuffer = VehicleConfigMsg_C_read(&configData->servicerVehicleConfigInMsg);
    debrisVehicleConfigInMsgBuffer = VehicleConfigMsg_C_read(&configData->debrisVehicleConfigInMsg);
    eForceInMsgBuffer = CmdForceInertialMsg_C_read(&configData->eForceInMsg);
    
    // - calculate control force
    calc_RelativeMotionControl(configData, servicerTransInMsgBuffer, debrisTransInMsgBuffer, servicerAttInMsgBuffer, servicerVehicleConfigInMsgBuffer, debrisVehicleConfigInMsgBuffer, eForceInMsgBuffer, &forceInertialOutMsgBuffer, &forceBodyOutMsgBuffer);
    
    // - write the module output messages
    CmdForceInertialMsg_C_write(&forceInertialOutMsgBuffer, &configData->forceInertialOutMsg, moduleID, callTime);
    CmdForceBodyMsg_C_write(&forceBodyOutMsgBuffer, &configData->forceBodyOutMsg, moduleID, callTime);

    return;
}

/*! This function calculates the control force of the Electrostatic Tractor Relative Motion Control based on
 current relative position and velocity, and desired relative position
 @return void
 @param configData The configuration data associated with the module
 @param servicerTransInMsgBuffer Servicer's position and velocity
 @param debrisTransInMsgBuffer Debris' position and velocity
 @param servicerAttInMsgBuffer Servicer's attitude
 @param servicerVehicleConfigInMsgBuffer Servicer Vehicle Configuration
 @param debrisVehicleConfigInMsgBuffer Servicer Vehicle Configuration
 @param eForceInMsgBuffer Electrostatic force on servicer
 @param forceInertialOutMsgBuffer inertial force output (3-axis)
 @param forceBodyOutMsgBuffer body force output (3-axis)
 */
void calc_RelativeMotionControl(etSphericalControlConfig *configData, NavTransMsgPayload servicerTransInMsgBuffer,
                                NavTransMsgPayload debrisTransInMsgBuffer, NavAttMsgPayload servicerAttInMsgBuffer,
                                VehicleConfigMsgPayload servicerVehicleConfigInMsgBuffer,
                                VehicleConfigMsgPayload debrisVehicleConfigInMsgBuffer,
                                CmdForceInertialMsgPayload eForceInMsgBuffer,
                                CmdForceInertialMsgPayload *forceInertialOutMsgBuffer,
                                CmdForceBodyMsgPayload *forceBodyOutMsgBuffer)
{
    // relative motion control according to "Relative Motion Control For Two-Spacecraft Electrostatic Orbit Corrections" https://doi.org/10.2514/1.56118
    
    // DCMs
    double dcm_HN[3][3]; // [HN] from inertial frame to Hill frame
    double dcm_NH[3][3]; // [NH] from Hill frame to inertial frame
    hillFrame(servicerTransInMsgBuffer.r_BN_N, servicerTransInMsgBuffer.v_BN_N, dcm_HN);
    m33Transpose(dcm_HN, dcm_NH);
    // relative position and velocity vector
    double rho_N[3]; // position
    double rho_H[3]; // in Hill frame components
    double rhoDot_N[3]; // inertial (absolute) relative velocity
    double rhoPrime_H[3]; // Hill relative velocity in Hill frame
    double r_TN_mag = v3Norm(servicerTransInMsgBuffer.r_BN_N); // magnitude of servicer position vector
    double omega_HN[3]; // angular velocity vector of Hill frame with respect to inertial frame
    v3Subtract(debrisTransInMsgBuffer.r_BN_N, servicerTransInMsgBuffer.r_BN_N, rho_N);
    m33MultV3(dcm_HN, rho_N, rho_H);
    v3Subtract(debrisTransInMsgBuffer.v_BN_N, servicerTransInMsgBuffer.v_BN_N, rhoDot_N);
    double rTN_cross_vTN[3]; // temporary vector
    double omega_cross_rho_N[3]; // temporary vector
    double rhoDot_m_rel[3]; // temporary vector
    v3Cross(servicerTransInMsgBuffer.r_BN_N, servicerTransInMsgBuffer.v_BN_N, rTN_cross_vTN);
    v3Scale(1./r_TN_mag/r_TN_mag, rTN_cross_vTN, omega_HN);
    v3Cross(omega_HN, rho_N, omega_cross_rho_N);
    v3Subtract(rhoDot_N, omega_cross_rho_N, rhoDot_m_rel);
    m33MultV3(dcm_HN, rhoDot_m_rel, rhoPrime_H);
    // cartesian coordinates
    double x = rho_H[0];
    double y = rho_H[1];
    double z = rho_H[2];
    // spherical coordinates
    double L = v3Norm(rho_H); // separation distance
    double theta = atan2(x,-y); // in-plane rotation angle
    double phi = safeAsin(-z/L); // out-of-plane rotation angle
    // more DCMs
    double dcm_SH[3][3]; // [SH] from Hill frame to spherical frame
    m33Set(cos(phi)*sin(theta), -cos(theta)*cos(phi), -sin(phi),
           cos(theta), sin(theta), 0.,
           sin(theta)*sin(phi), -cos(theta)*sin(phi), cos(phi),
           dcm_SH);
    double dcm_SN[3][3]; // [SN] from inertial frame to spherical frame
    double dcm_NS[3][3]; // [NS] from spherical frame to inertial frame
    double dcm_TN[3][3]; // [TN] from inertial frame to body frame of servicer
    m33MultM33(dcm_SH, dcm_HN, dcm_SN);
    m33Transpose(dcm_SN, dcm_NS);
    MRP2C(servicerAttInMsgBuffer.sigma_BN, dcm_TN);
    // state vector
    double X[3];
    v3Set(L, theta, phi, X);
    // state vector derivative
    double TransfMatrix[3][3]; // transformation matrix to map state vector derivative from Hill components to spherical components
    m33Set(cos(phi)*sin(theta), -cos(theta)*cos(phi), -sin(phi),
           cos(theta)/cos(phi)/L, 1./cos(phi)*sin(theta)/L, 0.,
           -sin(theta)*sin(phi)/L, cos(theta)*sin(phi)/L, -cos(phi)/L,
           TransfMatrix);
    double XDot[3]; // state vector derivative
    m33MultV3(TransfMatrix, rhoPrime_H, XDot);
    double LDot = XDot[0];
    double thetaDot = XDot[1];
    double phiDot = XDot[2];
    // control matrices [F] and [G]
    double mu = configData->mu; // [m^3/s^2] Earth's gravitational parameter
    classicElements elements;
    rv2elem(mu, servicerTransInMsgBuffer.r_BN_N, servicerTransInMsgBuffer.v_BN_N, &elements);
    double a = elements.a;
    double n = sqrt(mu/a/a/a); // mean motion
    
    double Fvector[3];
    v3Set(1./4.*L*(n*n*(-6.*cos(2.*theta)*cos(phi)*cos(phi)+5.*cos(2.*phi)+1.)+4.*thetaDot*cos(phi)*cos(phi)*(2.*n+thetaDot)+4.*phiDot*phiDot),
          (3.*n*n*sin(theta)*cos(theta)+2.*phiDot*tan(phi)*(n+thetaDot))-2.*LDot/L*(n+thetaDot),
          1./4.*sin(2.*phi)*(n*n*(3.*cos(2.*theta)-5.)-2.*thetaDot*(2.*n+thetaDot))-2.*LDot/L*phiDot,
          Fvector);
    double GmatrixInv[3][3]; // inverse of [G]
    m33Set(1., 0., 0.,
           0., L*cos(phi), 0.,
           0., 0., -L,
           GmatrixInv);
    // desired state vector
    double X_r[3];
    v3Set(configData->L_r,
          configData->theta_r,
          configData->phi_r,
          X_r);
    // feedback control acceleration
    double P_XDot[3]; // temporary vector
    double K_X[3]; // temporary vector
    double XmX_r[3]; // temporary vector
    double mP[3]; // temporary vector
    double PK[3]; // temporary vector
    double zeros[3]; // temporary vector
    double PKF[3]; // temporary vector
    v3SetZero(zeros);
    v3Subtract(X, X_r, XmX_r);
    m33MultV3(RECAST3X3 configData->P, XDot, P_XDot);
    m33MultV3(RECAST3X3 configData->K, XmX_r, K_X);
    v3Subtract(zeros, P_XDot, mP);
    v3Subtract(mP, K_X, PK);
    v3Subtract(PK, Fvector, PKF);
    double u_S[3]; // feedback control acceleration
    m33MultV3( GmatrixInv, PKF, u_S);   // u_S = [G]^-1*(-[P]*XDot-[K]*(X-X_r)-[F])
    
    // control thrust force
    double m_T; // mass of servicer
    double m_D; // mass of debris
    double FcmuTD[3]; // temporary vector
    double Fc_S[3]; // electrostatic force in spherical frame
    double uFc[3]; // temporary vector
    double T_S[3]; // control thrust force in spherical frame
    double T_N[3]; // control thrust force in inertial frame
    double T_T[3]; // control thrust force in servicer body frame
    m_T = servicerVehicleConfigInMsgBuffer.massSC;
    m_D = debrisVehicleConfigInMsgBuffer.massSC;
    double muTD = 1./m_T+1./m_D;
    m33MultV3(dcm_SN, eForceInMsgBuffer.forceRequestInertial, Fc_S);
    v3Scale(muTD, Fc_S, FcmuTD);
    v3Add(u_S, FcmuTD, uFc);
    v3Scale(-m_T, uFc, T_S);
    m33MultV3(dcm_NS, T_S, T_N);
    m33MultV3(dcm_TN, T_N, T_T);
    v3Copy(T_N, forceInertialOutMsgBuffer->forceRequestInertial);
    v3Copy(T_T, forceBodyOutMsgBuffer->forceRequestBody);
    
    return;
}
