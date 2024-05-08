/*
 ISC License

 Copyright (c) 2016-2017, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "fswAlgorithms/attDetermination/sunlineSEKF/sunlineSEKF.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for theCSS WLS estimator.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the CSS WLS estimator
 @param moduleID The module identifier
 */
void SelfInit_sunlineSEKF(sunlineSEKFConfig *configData, int64_t moduleID)
{
    NavAttMsg_C_init(&configData->navStateOutMsg);
    SunlineFilterMsg_C_init(&configData->filtDataOutMsg);
}


/*! This method resets the sunline attitude filter to an initial state and
 initializes the internal estimation matrices.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_sunlineSEKF(sunlineSEKFConfig *configData, uint64_t callTime,
                      int64_t moduleID)
{
    CSSConfigMsgPayload cssConfigInBuffer;

    /*! - Zero the local configuration data structures and outputs */
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();
    mSetZero(configData->cssNHat_B, MAX_NUM_CSS_SENSORS, 3);

    // check input messages are included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineSEKF.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineSEKF.cssDataInMsg wasn't connected.");
    }

    /*! - Read coarse sun sensor configuration information.*/
    cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "sunlineSEKF.cssConfigInMsg.nCSS must not be greater than "
                                                  "MAX_N_CSS_MEAS value.");
    }

    /*! - For each coarse sun sensor, convert the configuration data over from structure to body*/
    for(uint32_t i=0; i<cssConfigInBuffer.nCSS; i++)
    {
        v3Copy(cssConfigInBuffer.cssVals[i].nHat_B, &(configData->cssNHat_B[i*3]));
    }
    /*! - Save the count of sun sensors for later use */
    configData->numCSSTotal = cssConfigInBuffer.nCSS;
    
    /*! - Initialize filter parameters to max values */
    configData->timeTag = callTime*NANO2SEC;
    configData->dt = 0.0;
    configData->numStates = EKF_N_STATES_SWITCH;
    configData->numObs = MAX_N_CSS_MEAS;
    
    /*! Initalize the filter to use b_1 of the body frame to make frame*/
    v3Set(1, 0, 0, configData->bVec_B);
    configData->switchTresh = 0.866;
    
    /*! - Ensure that all internal filter matrices are zeroed*/
    vSetZero(configData->obs, configData->numObs);
    vSetZero(configData->yMeas, configData->numObs);
    vSetZero(configData->xBar, configData->numStates);
    mSetZero(configData->covarBar, configData->numStates, configData->numStates);
    
    mSetIdentity(configData->stateTransition, configData->numStates, configData->numStates);
    mSetIdentity(configData->W_BS, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);

    mSetZero(configData->dynMat, configData->numStates, configData->numStates);
    mSetZero(configData->measMat, configData->numObs, configData->numStates);
    mSetZero(configData->kalmanGain, configData->numStates, configData->numObs);
    
    mSetZero(configData->measNoise, configData->numObs, configData->numObs);
    mSetIdentity(configData->procNoise,  configData->numStates-3, configData->numStates-3);
    mScale(configData->qProcVal, configData->procNoise, configData->numStates-3, configData->numStates-3, configData->procNoise);
    
    return;
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_sunlineSEKF(sunlineSEKFConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double newTimeTag;
    double Hx[MAX_N_CSS_MEAS];
    double states_BN[EKF_N_STATES_SWITCH];
    double sunheading_hat[SKF_N_STATES_HALF];
    uint64_t timeOfMsgWritten;
    int isWritten;
    SunlineFilterMsgPayload sunlineDataOutBuffer;

    /*! - Read the input parsed CSS sensor data message*/
    configData->cssSensorInBuffer = CSSArraySensorMsg_C_read(&configData->cssDataInMsg);
    timeOfMsgWritten = CSSArraySensorMsg_C_timeWritten(&configData->cssDataInMsg);
    isWritten = CSSArraySensorMsg_C_isWritten(&configData->cssDataInMsg);

    v3Normalize(&configData->state[0], sunheading_hat);
    
    
    /*! - Check for switching frames */
    if (fabs(v3Dot(configData->bVec_B, sunheading_hat)) > configData->switchTresh)
    {
        sunlineSEKFSwitch(configData->bVec_B, configData->state, configData->covar);
    }
    
    /*! - If the time tag from the measured data is new compared to previous step, 
          propagate and update the filter*/
    newTimeTag = timeOfMsgWritten * NANO2SEC;
    if(newTimeTag >= configData->timeTag && isWritten)
    {
        sunlineTimeUpdate(configData, newTimeTag);
        sunlineMeasUpdate(configData, newTimeTag);
    }
    
    /*! - If current clock time is further ahead than the measured time, then
          propagate to this current time-step*/
    newTimeTag = callTime*NANO2SEC;
    if(newTimeTag > configData->timeTag)
    {
        sunlineTimeUpdate(configData, newTimeTag);
        vCopy(configData->xBar, EKF_N_STATES_SWITCH, configData->x);
        mCopy(configData->covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, configData->covar);
    }
    
    /* Compute post fit residuals once that data has been processed */
    mMultM(configData->measMat, configData->numObs, SKF_N_STATES, configData->x, SKF_N_STATES, 1, Hx);
    mSubtract(configData->yMeas, configData->numObs, 1, Hx, configData->postFits);
    
    /* Switch the rates back to omega_BN instead of oemga_SB */
    vCopy(configData->state, EKF_N_STATES_SWITCH, states_BN);
    vScale(-1, &(states_BN[3]), 2, &(states_BN[3]));
    
    /*! - Write the sunline estimate into the copy of the navigation message structure*/
	v3Copy(configData->state, configData->outputSunline.vehSunPntBdy);
    v3Normalize(configData->outputSunline.vehSunPntBdy,
        configData->outputSunline.vehSunPntBdy);
    configData->outputSunline.timeTag = configData->timeTag;
    NavAttMsg_C_write(&configData->outputSunline, &configData->navStateOutMsg, moduleID, callTime);

    /*! - Populate the filter states output buffer and write the output message*/
    sunlineDataOutBuffer.timeTag = configData->timeTag;
    sunlineDataOutBuffer.numObs = (int) configData->numObs;
    memmove(sunlineDataOutBuffer.covar, configData->covar,
            EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH*sizeof(double));
    memmove(sunlineDataOutBuffer.state, states_BN, EKF_N_STATES_SWITCH*sizeof(double));
    memmove(sunlineDataOutBuffer.stateError, configData->x, SKF_N_STATES*sizeof(double));
    memmove(sunlineDataOutBuffer.postFitRes, configData->postFits, MAX_N_CSS_MEAS*sizeof(double));
    SunlineFilterMsg_C_write(&sunlineDataOutBuffer, &configData->filtDataOutMsg, moduleID, callTime);

    return;
}

/*! This method performs the time update for the sunline kalman filter.
     It calls for the updated Dynamics Matrix, as well as the new states and STM.
     It then updates the covariance, with process noise.
	 @return void
     @param configData The configuration data associated with the CSS estimator
     @param updateTime The time that we need to fix the filter to (seconds)
*/
void sunlineTimeUpdate(sunlineSEKFConfig *configData, double updateTime)
{
    double stmT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], covPhiT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double Gamma[EKF_N_STATES_SWITCH][EKF_N_STATES_SWITCH - 3];
    double qGammaT[(EKF_N_STATES_SWITCH-3)*EKF_N_STATES_SWITCH], gammaQGammaT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double d_tilde[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double dcm_BS[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    mSetZero(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    
	/*! compute timne step */
	configData->dt = updateTime - configData->timeTag;
    
    /*! - Propagate the previous reference states and STM to the current time */
    sunlineDynMatrix(configData->state, configData->bVec_B, configData->dt, configData->dynMat);
    sunlineStateSTMProp(configData->dynMat, configData->bVec_B, configData->dt, configData->state, configData->stateTransition);

    /* Do the time update on the state error */
    mMultV(configData->stateTransition, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, configData->x, configData->xBar);
    
    /*! - Update the covariance */
    /*Pbar = Phi*P*Phi^T + Gamma*Q*Gamma^T*/
    mTranspose(configData->stateTransition, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, stmT);
    mMultM(configData->covar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, stmT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covPhiT);
    mMultM(configData->stateTransition, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covPhiT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, configData->covarBar);
    
    sunlineSEKFComputeDCM_BS(configData->state, configData->bVec_B, &dcm_BS[0][0]);
    /*Compute Gamma and add gammaQGamma^T to Pbar. This is the process noise addition*/
    mSetIdentity(d_tilde, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mScale(configData->dt, d_tilde, SKF_N_STATES_HALF, SKF_N_STATES_HALF, d_tilde);
    mSetSubMatrix(&(d_tilde[0][0]), 1, 2, Gamma, 5, 2, 3, 0);
    mSetSubMatrix(&(d_tilde[1][0]), 1, 2, Gamma, 5, 2, 4, 0);
    v3Tilde(configData->state, d_tilde);
    mMultM(d_tilde, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF, d_tilde);
    mScale(configData->dt*configData->dt/2, d_tilde, SKF_N_STATES_HALF, SKF_N_STATES_HALF, d_tilde);
    mSetSubMatrix(&(d_tilde[0][1]), 1, 2, Gamma, 5, 2, 0, 0);
    mSetSubMatrix(&(d_tilde[1][1]), 1, 2, Gamma, 5, 2, 1, 0);
    mSetSubMatrix(&(d_tilde[2][1]), 1, 2, Gamma, 5, 2, 2, 0);
    
    mMultMt(configData->procNoise, (EKF_N_STATES_SWITCH-3),(EKF_N_STATES_SWITCH-3), Gamma, EKF_N_STATES_SWITCH, (EKF_N_STATES_SWITCH-3), qGammaT);
    mMultM(Gamma, EKF_N_STATES_SWITCH,(EKF_N_STATES_SWITCH-3), qGammaT, (EKF_N_STATES_SWITCH-3), EKF_N_STATES_SWITCH, gammaQGammaT);
    mAdd(configData->covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, gammaQGammaT, configData->covarBar);
    
	configData->timeTag = updateTime;
}


/*! This method propagates a sunline state vector forward in time.  Note
 that the calling parameter is updated in place to save on data copies.
 This also updates the STM using the dynamics matrix.
	@return void
	@param stateInOut
    @param dynMat
    @param bVec
    @param dt time step
    @param stateTransition
 */
void sunlineStateSTMProp(double dynMat[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double bVec[SKF_N_STATES], double dt, double *stateInOut, double *stateTransition)
{
    
    double deltatASTM[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double propagatedVel[SKF_N_STATES_HALF];
    double omegaCrossd[SKF_N_STATES_HALF];
    double omega_BN_S[SKF_N_STATES_HALF] = {0, -stateInOut[3], -stateInOut[4]};
    double omega_BN_B[SKF_N_STATES_HALF];
    double dcm_BS[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    
    mSetZero(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    
    sunlineSEKFComputeDCM_BS(stateInOut, bVec, &dcm_BS[0][0]);
    mMultV(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF, omega_BN_S, omega_BN_B);
    /* Set local variables to zero*/
    vSetZero(propagatedVel, SKF_N_STATES_HALF);
    
    /*! Begin state update steps */
    /*! Take omega cross d*/
    v3Cross(omega_BN_B, stateInOut, omegaCrossd);
    
    /*! - Multiply omega cross d by dt and add to state to propagate */
    v3Scale(-dt, omegaCrossd, propagatedVel);
    v3Add(stateInOut, propagatedVel, stateInOut);
    
    /*! Begin STM propagation step */
    mSetIdentity(stateTransition, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mScale(dt, dynMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, deltatASTM);
    mAdd(stateTransition, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, deltatASTM, stateTransition);
    
    return;
}

/*! This method computes the dynamics matrix, which is the derivative of the
 dynamics F by the state X, evaluated at the reference state. It takes in the
 configure data and updates this A matrix pointer called dynMat
 @return void
 @param states Updated states
 @param bVec b vector
 @param dt Time step
 @param dynMat Pointer to the Dynamic Matrix
 */

void sunlineDynMatrix(double states[EKF_N_STATES_SWITCH], double bVec[SKF_N_STATES], double dt, double *dynMat)
{
    double skewOmega[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double skewStates[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double omega_BN_S[SKF_N_STATES_HALF] = {0, -states[3], -states[4]};
    double omega_BN_B[SKF_N_STATES_HALF];
    double dcm_BS[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    
    sunlineSEKFComputeDCM_BS(states, bVec, &dcm_BS[0][0]);
    mMultV(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF, omega_BN_S, omega_BN_B);

    mSetZero(skewOmega, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    v3Tilde(omega_BN_B, skewOmega);
    m33Scale(-1, &skewOmega[0], &skewOmega[0]); // bring to omega_SB with negative sign
    v3Tilde(states, skewStates);
    m33Scale(-1, &skewStates[0], &skewStates[0]); // bring to omega_SB with negative sign
    mMultM(skewStates, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF, skewStates);
    
    /* - omega_tilde in dynamics */
    mSetSubMatrix(skewOmega, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dynMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, 0, 0);
    
    /* Populate the first 3x3 matrix of the dynamics matrix*/
    mTranspose(dynMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, dynMat);
    mTranspose(skewStates, SKF_N_STATES_HALF, SKF_N_STATES_HALF, skewStates);
    mSetSubMatrix(&(skewStates[1][0]), 2, 3, dynMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, 3, 0);

    mTranspose(dynMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, dynMat);

    return;
}


/*! This method performs the measurement update for the sunline kalman filter.
 It applies the observations in the obs vectors to the current state estimate and
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param updateTime The time that we need to fix the filter to (seconds)
 */
void sunlineMeasUpdate(sunlineSEKFConfig *configData, double updateTime)
{
    /*! - Compute the valid observations and the measurement model for all observations*/
    int numObsInt = (int) configData->numObs;
    sunlineHMatrixYMeas(configData->state, configData->numCSSTotal, configData->cssSensorInBuffer.CosValue, configData->sensorUseThresh, configData->cssNHat_B, configData->obs, configData->yMeas, &(numObsInt), configData->measMat);
    configData->numObs = (size_t) numObsInt;
    
    /*! - Compute the Kalman Gain. */
    sunlineKalmanGain(configData->covarBar, configData->measMat, configData->qObsVal, configData->numObs, configData->kalmanGain);
    
    /* Logic to switch from EKF to CKF. If the covariance is too large, switching references through an EKF could lead to filter divergence in extreme cases. In order to remedy this, past a certain infinite norm of the covariance, we update with a CKF in order to bring down the covariance. */
    
    if (vMaxAbs(configData->covar, EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH) > configData->eKFSwitch){
    /*! - Compute the update with a CKF */
    sunlineCKFUpdate(configData->xBar, configData->kalmanGain, configData->covarBar, configData->qObsVal, configData->numObs, configData->yMeas, configData->measMat, configData->x,configData->covar);
    }
    else{
    /* - Compute the update with a EKF, notice the reference state is added as an argument because it is changed by the filter update */
    sunlineSEKFUpdate(configData->kalmanGain, configData->covarBar, configData->qObsVal, configData->numObs, configData->yMeas, configData->measMat, configData->state, configData->x, configData->covar);
    }
}

/*! This method computes the updated with a Classical Kalman Filter
 @return void
 @param xBar The state after a time update
 @param kalmanGain The computed Kalman Gain
 @param covarBar The time updated covariance
 @param qObsVal The observation noise
 @param numObs The amount of CSSs that get measurements
 @param yObs The y vector after receiving the measurements
 @param hObs The H matrix filled with the observations
 @param x Pointer to the state error for modification
 @param covar Pointer to the covariance after update
 */

void sunlineCKFUpdate(double xBar[EKF_N_STATES_SWITCH], double kalmanGain[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS], double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double *x, double *covar)
{
    double measMatx[MAX_N_CSS_MEAS], innov[MAX_N_CSS_MEAS], kInnov[EKF_N_STATES_SWITCH];
    double eye[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], kH[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double eyeKalH[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], eyeKalHT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double eyeKalHCovarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], kalR[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], kalRKalT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Set variables to zero */
    mSetZero(kH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(eyeKalHT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(kalRKalT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(kalT, MAX_N_CSS_MEAS, EKF_N_STATES_SWITCH);
    mSetZero(kalR, EKF_N_STATES_SWITCH, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, numObs, numObs);
    mScale(qObsVal, noiseMat, numObs, numObs, noiseMat);
    
    /*! - Compute innovation, multiply it my Kalman Gain, and add it to xBar*/
    mMultM(hObs, numObs, EKF_N_STATES_SWITCH, xBar, EKF_N_STATES_SWITCH, 1, measMatx);
    vSubtract(yObs, numObs, measMatx, innov);
    mMultM(kalmanGain, EKF_N_STATES_SWITCH, numObs, innov, numObs, 1, kInnov);
    vAdd(xBar, EKF_N_STATES_SWITCH, kInnov, x);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, EKF_N_STATES_SWITCH, numObs, hObs, numObs, EKF_N_STATES_SWITCH, kH);
    mSetIdentity(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSubtract(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, kH, eyeKalH);
    mTranspose(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHT);
    mMultM(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, EKF_N_STATES_SWITCH, numObs, noiseMat, numObs, numObs, kalR);
    mTranspose(kalmanGain, EKF_N_STATES_SWITCH, numObs, kalT);
    mMultM(kalR, EKF_N_STATES_SWITCH, numObs, kalT, numObs, EKF_N_STATES_SWITCH, kalRKalT);
    mAdd(covar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, kalRKalT, covar);
    
    
}

/*! This method computes the updated with a Extended Kalman Filter
 @return void
 @param kalmanGain The computed Kalman Gain
 @param covarBar The time updated covariance
 @param qObsVal The observation noise
 @param numObs The amount of CSSs that get measurements
 @param yObs The y vector after receiving the measurements
 @param hObs The H matrix filled with the observations
 @param states Pointer to the states
 @param x Pointer to the state error for modification
 @param covar Pointer to the covariance after update
 */
void sunlineSEKFUpdate(double kalmanGain[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS], double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double *states, double *x, double *covar)
{

    double eye[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], kH[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double eyeKalH[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], eyeKalHT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double eyeKalHCovarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], kalR[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], kalRKalT[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Set variables to zero */
    mSetZero(kH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(eyeKalHT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(kalRKalT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetZero(kalT, MAX_N_CSS_MEAS, EKF_N_STATES_SWITCH);
    mSetZero(kalR, EKF_N_STATES_SWITCH, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, numObs, numObs);
    mScale(qObsVal, noiseMat, numObs, numObs, noiseMat);
    
    /*! - Update the state error*/
    mMultV(kalmanGain, EKF_N_STATES_SWITCH, numObs, yObs, x);

    /*! - Change the reference state*/
    vAdd(states, EKF_N_STATES_SWITCH, x, states);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, EKF_N_STATES_SWITCH, numObs, hObs, numObs, EKF_N_STATES_SWITCH, kH);
    mSetIdentity(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSubtract(eye, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, kH, eyeKalH);
    mTranspose(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHT);
    mMultM(eyeKalH, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, eyeKalHT, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, EKF_N_STATES_SWITCH, numObs, noiseMat, numObs, numObs, kalR);
    mTranspose(kalmanGain, EKF_N_STATES_SWITCH, numObs, kalT);
    mMultM(kalR, EKF_N_STATES_SWITCH, numObs, kalT, numObs, EKF_N_STATES_SWITCH, kalRKalT);
    mAdd(covar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, kalRKalT, covar);
    
}

/*! This method computes the H matrix, defined by dGdX. As well as computing the 
 innovation, difference between the measurements and the expected measurements.
 This methods modifies the numObs, measMat, and yMeas. 
 @return void
 @param states
 @param numCSS The total number of CSS
 @param cssSensorCos The list of the measurements from the CSSs
 @param sensorUseThresh Thresh The Threshold below which the measuremnts are not read
 @param cssNHat_B The normals vector for each of the CSSs
 @param obs Pointer to the observations
 @param yMeas Pointer to the innovation
 @param numObs Pointer to the number of observations
 @param measMat Point to the H measurement matrix
 */

void sunlineHMatrixYMeas(double states[EKF_N_STATES_SWITCH], size_t numCSS, double cssSensorCos[MAX_N_CSS_MEAS], double sensorUseThresh, double cssNHat_B[MAX_NUM_CSS_SENSORS*3], double *obs, double *yMeas, int *numObs, double *measMat)
{
    uint32_t i, obsCounter;
    double sensorNormal[3];
    
    v3SetZero(sensorNormal);

    obsCounter = 0;
    /*! - Loop over all available coarse sun sensors and only use ones that meet validity threshold*/
    for(i=0; i<numCSS; i++)
    {
        if(cssSensorCos[i] > sensorUseThresh)
        {
            /*! - For each valid measurement, copy observation value and compute expected obs value and fill out H matrix.*/
            v3Copy(&(cssNHat_B[i*3]), sensorNormal);
            
            *(obs+obsCounter) = cssSensorCos[i];
            *(yMeas+obsCounter) = cssSensorCos[i] - v3Dot(&(states[0]), sensorNormal);
            mSetSubMatrix(&(cssNHat_B[i*3]), 1, 3, measMat, MAX_NUM_CSS_SENSORS, EKF_N_STATES_SWITCH, obsCounter, 0);
            obsCounter++;
        }
    }
    *numObs = (int) obsCounter;
}



/*! This method computes the Kalman gain given the measurements.
 @return void
 @param covarBar The time updated covariance
 @param hObs The H matrix filled with the observations
 @param qObsVal The observation noise
 @param numObs The number of observations
 @param kalmanGain Pointer to the Kalman Gain
 */

void sunlineKalmanGain(double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double *kalmanGain)
{
    double hObsT[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    double covHT[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    double hCovar[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], hCovarHT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double rMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Setting all local variables to zero */
    mSetZero(hObsT, EKF_N_STATES_SWITCH, MAX_N_CSS_MEAS);
    mSetZero(covHT, EKF_N_STATES_SWITCH, MAX_N_CSS_MEAS);
    mSetZero(hCovar, MAX_N_CSS_MEAS, EKF_N_STATES_SWITCH);
    mSetZero(hCovarHT, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(rMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    
    mTranspose(hObs, numObs, EKF_N_STATES_SWITCH, hObsT);
    
    mMultM(covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, hObsT, EKF_N_STATES_SWITCH, numObs, covHT);
    mMultM(hObs, numObs, EKF_N_STATES_SWITCH, covarBar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, hCovar);
    mMultM(hCovar, numObs, EKF_N_STATES_SWITCH, hObsT, EKF_N_STATES_SWITCH, numObs, hCovarHT);
    
    mSetIdentity(rMat, numObs, numObs);
    mScale(qObsVal, rMat, numObs, numObs, rMat);
    
    /*! - Add measurement noise */
    mAdd(hCovarHT, numObs, numObs, rMat, hCovarHT);
    
    /*! - Invert the previous matrix */
    mInverse(hCovarHT, numObs, hCovarHT);
    
    /*! - Compute the Kalman Gain */
    mMultM(covHT, EKF_N_STATES_SWITCH, numObs, hCovarHT, numObs, numObs, kalmanGain);
    
}


/*! This method computes the dcms necessary for the switch between the two frames.
 It the switches the states and the covariance, and sets s2 to be the new, different vector of the body frame.
 @return void
 @param bVec_B Pointer to b vector
 @param states Pointer to the states
 @param covar Pointer to the covariance
 */

void sunlineSEKFSwitch(double *bVec_B, double *states, double *covar)
{
    double dcm_BSold[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double dcm_BSnew_T[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double dcm_SnewSold[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double switchMatP[EKF_N_STATES_SWITCH][EKF_N_STATES_SWITCH];
    double switchMat[EKF_N_STATES_SWITCH][EKF_N_STATES_SWITCH];
    
    double sun_heading_norm[SKF_N_STATES_HALF];
    double b1[SKF_N_STATES_HALF];
    double b2[SKF_N_STATES_HALF];
    
    /*!  Set the body frame vectors*/
    v3Set(1, 0, 0, b1);
    v3Set(0, 1, 0, b2);
    v3Normalize(&(states[0]), sun_heading_norm);
    
    /*! Populate the dcm_BS with the "old" S-frame*/
    sunlineSEKFComputeDCM_BS(sun_heading_norm, bVec_B, &dcm_BSold[0][0]);
    
    if (v3IsEqual(bVec_B, b1, 1e-10))
    {
        sunlineSEKFComputeDCM_BS(sun_heading_norm, b2, &dcm_BSnew_T[0][0]);
        v3Copy(b2, bVec_B);
    }
    else
    {
        sunlineSEKFComputeDCM_BS(sun_heading_norm, b1, &dcm_BSnew_T[0][0]);
        v3Copy(b1, bVec_B);
    }
    
    mTranspose(dcm_BSnew_T, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dcm_BSnew_T);
    mMultM(dcm_BSnew_T, 3, 3, dcm_BSold, 3, 3, dcm_SnewSold);
    
    mSetIdentity(switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH);
    mSetSubMatrix(&dcm_SnewSold[1][1], 1, 2, &switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, 3, 3);
    mSetSubMatrix(&dcm_SnewSold[2][1], 1, 2, &switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, 4, 3);
    
    mMultV(switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, states, states);
    mMultM(switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covar, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, switchMatP);
    mTranspose(switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, switchMat);
    mMultM(switchMatP, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, switchMat, EKF_N_STATES_SWITCH, EKF_N_STATES_SWITCH, covar);
    return;
}

/*! @brief compute the DCM
 @param sunheading array of sun heading measurement states
 @param bVec    array of bVec values
 @param dcm DCM pointer to be returned
 @return void
 */
void sunlineSEKFComputeDCM_BS(double sunheading[SKF_N_STATES_HALF], double bVec[SKF_N_STATES_HALF], double *dcm){
    double s1_B[SKF_N_STATES_HALF];
    double s2_B[SKF_N_STATES_HALF];
    double s3_B[SKF_N_STATES_HALF];
    
    mSetZero(dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    v3SetZero(s2_B);
    v3SetZero(s3_B);
    
    v3Normalize(sunheading, s1_B);
    v3Cross(sunheading, bVec, s2_B);
    if (v3Norm(s2_B) < 1E-5){
        mSetIdentity(dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    }
    else{
    v3Normalize(s2_B, s2_B);
    /*! Populate the dcm_BS with the "new" S-frame*/
    mSetSubMatrix(s1_B, 1, SKF_N_STATES_HALF, dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF, 0, 0);
    mSetSubMatrix(&(s2_B), 1, SKF_N_STATES_HALF, dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF, 1, 0);
    v3Cross(sunheading, s2_B, s3_B);
    v3Normalize(s3_B, s3_B);
    mSetSubMatrix(&(s3_B), 1, SKF_N_STATES_HALF, dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF, 2, 0);
    mTranspose(dcm, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dcm);
    }
    
}
