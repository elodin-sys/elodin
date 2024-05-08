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

#include "fswAlgorithms/attDetermination/sunlineEKF/sunlineEKF.h"
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
void SelfInit_sunlineEKF(sunlineEKFConfig *configData, int64_t moduleID)
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
void Reset_sunlineEKF(sunlineEKFConfig *configData, uint64_t callTime,
                      int64_t moduleID)
{
    
    CSSConfigMsgPayload cssConfigInBuffer;
   
    /*! - Zero the local configuration data structures and outputs */
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();
    mSetZero(configData->cssNHat_B, MAX_NUM_CSS_SENSORS, 3);

    // check if the required input messages are included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineEKF.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineEKF.cssDataInMsg wasn't connected.");
    }

    /*! - Read in mass properties and coarse sun sensor configuration information.*/
    cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "sunlineEKF.cssConfigInMsg.nCSS must not be greater than "
                                                  "MAX_N_CSS_MEAS value.");
    }

    /*! - For each coarse sun sensor, convert the configuration data over from structure to body*/
    for(uint32_t i=0; i<cssConfigInBuffer.nCSS; i++)
    {
        v3Copy(cssConfigInBuffer.cssVals[i].nHat_B, &(configData->cssNHat_B[i*3]));
        configData->CBias[i] = cssConfigInBuffer.cssVals[i].CBias;
    }
    /*! - Save the count of sun sensors for later use */
    configData->numCSSTotal = cssConfigInBuffer.nCSS;
    
    /*! - Initialize filter parameters to max values */
    configData->timeTag = callTime*NANO2SEC;
    configData->dt = 0.0;
    configData->numStates = SKF_N_STATES;
    configData->numObs = MAX_N_CSS_MEAS;
    
    /*! - Ensure that all internal filter matrices are zeroed*/
    vSetZero(configData->obs, (size_t) configData->numObs);
    vSetZero(configData->yMeas, (size_t) configData->numObs);
    vSetZero(configData->xBar, configData->numStates);
    mSetZero(configData->covarBar, configData->numStates, configData->numStates);
    
    mSetIdentity(configData->stateTransition, configData->numStates, configData->numStates);

    mSetZero(configData->dynMat, configData->numStates, configData->numStates);
    mSetZero(configData->measMat, (size_t) configData->numObs, configData->numStates);
    mSetZero(configData->kalmanGain, configData->numStates, (size_t) configData->numObs);
    
    mSetZero(configData->measNoise, (size_t) configData->numObs, (size_t) configData->numObs);
    mSetIdentity(configData->procNoise,  configData->numStates/2, configData->numStates/2);
    mScale(configData->qProcVal, configData->procNoise, configData->numStates/2, configData->numStates/2, configData->procNoise);
    
    return;
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_sunlineEKF(sunlineEKFConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double newTimeTag;
    double Hx[MAX_N_CSS_MEAS];
    uint64_t timeOfMsgWritten;
    int isWritten;
    SunlineFilterMsgPayload sunlineDataOutBuffer;

    /*! - Read the input parsed CSS sensor data message*/
    configData->cssSensorInBuffer = CSSArraySensorMsg_C_read(&configData->cssDataInMsg);
    timeOfMsgWritten = CSSArraySensorMsg_C_timeWritten(&configData->cssDataInMsg);
    isWritten = CSSArraySensorMsg_C_isWritten(&configData->cssDataInMsg);

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
        vCopy(configData->xBar, SKF_N_STATES, configData->x);
        mCopy(configData->covarBar, SKF_N_STATES, SKF_N_STATES, configData->covar);
    }
    
    /* Compute post fit residuals once that data has been processed */
    mMultM(configData->measMat, (size_t) configData->numObs, SKF_N_STATES, configData->x, SKF_N_STATES, 1, Hx);
    mSubtract(configData->yMeas, (size_t) configData->numObs, 1, Hx, configData->postFits);
    
    /*! - Write the sunline estimate into the copy of the navigation message structure*/
	v3Copy(configData->state, configData->outputSunline.vehSunPntBdy);
    v3Normalize(configData->outputSunline.vehSunPntBdy,
        configData->outputSunline.vehSunPntBdy);
    configData->outputSunline.timeTag = configData->timeTag;
    NavAttMsg_C_write(&configData->outputSunline, &configData->navStateOutMsg, moduleID, callTime);

    /*! - Populate the filter states output buffer and write the output message*/
    sunlineDataOutBuffer.timeTag = configData->timeTag;
    sunlineDataOutBuffer.numObs = configData->numObs;
    memmove(sunlineDataOutBuffer.covar, configData->covar,
            SKF_N_STATES*SKF_N_STATES*sizeof(double));
    memmove(sunlineDataOutBuffer.state, configData->state, SKF_N_STATES*sizeof(double));
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
void sunlineTimeUpdate(sunlineEKFConfig *configData, double updateTime)
{
    double stmT[SKF_N_STATES*SKF_N_STATES], covPhiT[SKF_N_STATES*SKF_N_STATES];
    double Gamma[SKF_N_STATES][SKF_N_STATES_HALF];
    double qGammaT[SKF_N_STATES_HALF*SKF_N_STATES], gammaQGammaT[SKF_N_STATES*SKF_N_STATES];
    double Id[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    
	configData->dt = updateTime - configData->timeTag;
    
    /*! - Propagate the previous reference states and STM to the current time */
    sunlineDynMatrix(configData->state, configData->dt, configData->dynMat);
    sunlineStateSTMProp(configData->dynMat, configData->dt, configData->state, configData->stateTransition);

    /* xbar = Phi*x */
    mMultV(configData->stateTransition, SKF_N_STATES, SKF_N_STATES, configData->x, configData->xBar);
    
    /*! - Update the covariance */
    /*Pbar = Phi*P*Phi^T + Gamma*Q*Gamma^T*/
    mTranspose(configData->stateTransition, SKF_N_STATES, SKF_N_STATES, stmT);
    mMultM(configData->covar, SKF_N_STATES, SKF_N_STATES, stmT, SKF_N_STATES, SKF_N_STATES, covPhiT);
    mMultM(configData->stateTransition, SKF_N_STATES, SKF_N_STATES, covPhiT, SKF_N_STATES, SKF_N_STATES, configData->covarBar);
    
    /*Compute Gamma and add gammaQGamma^T to Pbar. This is the process noise addition*/
//    double Gamma[6][3]={{configData->dt*configData->dt/2,0,0},{0,configData->dt*configData->dt/2,0},{0,0,configData->dt*configData->dt/2},{configData->dt,0,0},{0,configData->dt,0},{0,0,configData->dt}};
    mSetIdentity(Id, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mScale(configData->dt, Id, SKF_N_STATES_HALF, SKF_N_STATES_HALF, Id);
    mSetSubMatrix(Id, 3, 3, Gamma, 6, 3, 3, 0);
    mScale(configData->dt/2, Id, SKF_N_STATES_HALF, SKF_N_STATES_HALF, Id);
    mSetSubMatrix(Id, 3, 3, Gamma, 6, 3, 0, 0);

    mMultMt(configData->procNoise, SKF_N_STATES_HALF, SKF_N_STATES_HALF, Gamma, SKF_N_STATES, SKF_N_STATES_HALF, qGammaT);
    mMultM(Gamma, SKF_N_STATES, SKF_N_STATES_HALF, qGammaT, SKF_N_STATES_HALF, SKF_N_STATES, gammaQGammaT);
    mAdd(configData->covarBar, SKF_N_STATES, SKF_N_STATES, gammaQGammaT, configData->covarBar);
    
	configData->timeTag = updateTime;
}


/*! This method propagates a sunline state vector forward in time.  Note
    that the calling parameter is updated in place to save on data copies.
    This also updates the STM using the dynamics matrix.
	@return void
    @param dynMat
    @param dt
    @param stateInOut
    @param stateTransition
 */
void sunlineStateSTMProp(double dynMat[SKF_N_STATES*SKF_N_STATES], double dt, double *stateInOut, double *stateTransition)
{
    
    double propagatedVel[SKF_N_STATES_HALF];
    double pointUnit[SKF_N_STATES_HALF];
    double unitComp;
    double deltatASTM[SKF_N_STATES*SKF_N_STATES];
    
    /* Set local variables to zero*/
    mSetZero(deltatASTM, SKF_N_STATES, SKF_N_STATES);
    unitComp=0.0;
    vSetZero(pointUnit, SKF_N_STATES_HALF);
    vSetZero(propagatedVel, SKF_N_STATES_HALF);
    
    /*! Begin state update steps */
    /*! - Unitize the current estimate to find direction to restrict motion*/
    v3Normalize(stateInOut, pointUnit);
    unitComp = v3Dot(&(stateInOut[3]), pointUnit);
    v3Scale(unitComp, pointUnit, pointUnit);
    /*! - Subtract out rotation in the sunline axis because that is not observable
     for coarse sun sensors*/
    v3Subtract(&(stateInOut[3]), pointUnit, &(stateInOut[3]));
    v3Scale(dt, &(stateInOut[3]), propagatedVel);
    v3Add(stateInOut, propagatedVel, stateInOut);
    
    /*! Begin STM propagation step */
    mSetIdentity(stateTransition, SKF_N_STATES, SKF_N_STATES);
    mScale(dt, dynMat, SKF_N_STATES, SKF_N_STATES, deltatASTM);
    mAdd(stateTransition, SKF_N_STATES, SKF_N_STATES, deltatASTM, stateTransition);
    
    return;
}

/*! This method computes the dynamics matrix, which is the derivative of the
 dynamics F by the state X, evaluated at the reference state. It takes in the
 configure data and updates this A matrix pointer called dynMat
 @return void
 @param states Updated states
 @param dt Time step
 @param dynMat Pointer to the Dynamic Matrix
 */

void sunlineDynMatrix(double states[SKF_N_STATES], double dt, double *dynMat)
{
    double dddot, ddtnorm2[3][3];
    double I3[3][3], d2I3[3][3];
    double douterddot[3][3], douterd[3][3], neg2dd[3][3];
    double secondterm[3][3], firstterm[3][3];
    double normd2;
    double dFdd[3][3], dFdddot[3][3];
    
    /* dF1dd */
    mSetIdentity(I3, 3, 3);
    dddot = v3Dot(&(states[0]), &(states[3]));
    normd2 = v3Norm(&(states[0]))*v3Norm(&(states[0]));
    
    mScale(normd2, I3, 3, 3, d2I3);
    
    v3OuterProduct(&(states[0]), &(states[3]), douterddot);
    v3OuterProduct(&(states[0]), &(states[0]), douterd);
    
    m33Scale(-2.0, douterd, neg2dd);
    m33Add(d2I3, neg2dd, secondterm);
    m33Scale(dddot/(normd2*normd2), secondterm, secondterm);
    
    m33Scale(1.0/normd2, douterddot, firstterm);
    
    m33Add(firstterm, secondterm, dFdd);
    m33Scale(-1.0, dFdd, dFdd);
    
    /* Populate the first 3x3 matrix of the dynamics matrix*/
    mSetSubMatrix(dFdd, 3, 3, dynMat, SKF_N_STATES, SKF_N_STATES, 0, 0);
    
    /* dF1dddot */
    m33Scale(-1.0/normd2, douterd, ddtnorm2);
    m33Add(I3, ddtnorm2, dFdddot);
    
    /* Populate the second 3x3 matrix */
    mSetSubMatrix(dFdddot, 3, 3, dynMat, SKF_N_STATES, SKF_N_STATES, 0, 3);
    
    /* Only propagate d_dot if dt is greater than zero, if not leave dynMat zeroed*/
    if (dt>1E-10){
        /* dF2dd */
        m33Scale(1.0/dt, dFdd, dFdd);
        /* Populate the third 3x3 matrix of the dynamics matrix*/
        mSetSubMatrix(dFdd, 3, 3, dynMat, SKF_N_STATES, SKF_N_STATES, 3, 0);
    
        /* dF2dddot */
        m33Subtract(dFdddot, I3, dFdddot);
        m33Scale(1.0/dt, dFdddot, dFdddot);
        /* Populate the fourth 3x3 matrix */
        mSetSubMatrix(dFdddot, 3, 3, dynMat, SKF_N_STATES, SKF_N_STATES, 3, 3);
    }
    return;
}


/*! This method performs the measurement update for the sunline kalman filter.
 It applies the observations in the obs vectors to the current state estimate and
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param updateTime The time that we need to fix the filter to (seconds)
 */
void sunlineMeasUpdate(sunlineEKFConfig *configData, double updateTime)
{
    /*! - Compute the valid observations and the measurement model for all observations*/
    sunlineHMatrixYMeas(configData->state, (int) configData->numCSSTotal, configData->cssSensorInBuffer.CosValue, configData->sensorUseThresh, configData->cssNHat_B, configData->CBias, configData->obs, configData->yMeas, &(configData->numObs), configData->measMat);
    
    /*! - Compute the Kalman Gain. */
    sunlineKalmanGain(configData->covarBar, configData->measMat, configData->qObsVal, configData->numObs, configData->kalmanGain);
    
    /* Logic to switch from EKF to CKF. If the covariance is too large, switching references through an EKF could lead to filter divergence in extreme cases. In order to remedy this, past a certain infinite norm of the covariance, we update with a CKF in order to bring down the covariance. */
    
    if (vMaxAbs(configData->covar, SKF_N_STATES*SKF_N_STATES) > configData->eKFSwitch){
    /*! - Compute the update with a CKF */
    sunlineCKFUpdate(configData->xBar, configData->kalmanGain, configData->covarBar, configData->qObsVal, configData->numObs, configData->yMeas, configData->measMat, configData->x,configData->covar);
    }
    else{
    /*! - Compute the update with a EKF, notice the reference state is added as an argument because it is changed by the filter update */
    sunlineEKFUpdate(configData->kalmanGain, configData->covarBar, configData->qObsVal, configData->numObs, configData->yMeas, configData->measMat, configData->state, configData->x, configData->covar);
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

void sunlineCKFUpdate(double xBar[SKF_N_STATES], double kalmanGain[SKF_N_STATES*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES*SKF_N_STATES], double qObsVal, int numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double *x, double *covar)
{
    double measMatx[MAX_N_CSS_MEAS], innov[MAX_N_CSS_MEAS], kInnov[SKF_N_STATES];
    double eye[SKF_N_STATES*SKF_N_STATES], kH[SKF_N_STATES*SKF_N_STATES];
    double eyeKalH[SKF_N_STATES*SKF_N_STATES], eyeKalHT[SKF_N_STATES*SKF_N_STATES];
    double eyeKalHCovarBar[SKF_N_STATES*SKF_N_STATES], kalR[SKF_N_STATES*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*SKF_N_STATES], kalRKalT[SKF_N_STATES*SKF_N_STATES];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Set variables to zero */
    mSetZero(kH, SKF_N_STATES, SKF_N_STATES);
    mSetZero(eyeKalH, SKF_N_STATES, SKF_N_STATES);
    mSetZero(eyeKalHT, SKF_N_STATES, SKF_N_STATES);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, SKF_N_STATES, SKF_N_STATES);
    mSetZero(kalRKalT, SKF_N_STATES, SKF_N_STATES);
    mSetZero(kalT, MAX_N_CSS_MEAS, SKF_N_STATES);
    mSetZero(kalR, SKF_N_STATES, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, SKF_N_STATES, SKF_N_STATES);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, (size_t) numObs, (size_t) numObs);
    mScale(qObsVal, noiseMat, (size_t) numObs, (size_t) numObs, noiseMat);
    
    /*! - Compute innovation, multiply it my Kalman Gain, and add it to xBar*/
    mMultM(hObs, (size_t) numObs, SKF_N_STATES, xBar, SKF_N_STATES, 1, measMatx);
    vSubtract(yObs, (size_t) numObs, measMatx, innov);
    mMultM(kalmanGain, SKF_N_STATES, (size_t) numObs, innov, (size_t) numObs, 1, kInnov);
    vAdd(xBar, SKF_N_STATES, kInnov, x);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, SKF_N_STATES, (size_t) numObs, hObs, (size_t) numObs, SKF_N_STATES, kH);
    mSetIdentity(eye, SKF_N_STATES, SKF_N_STATES);
    mSubtract(eye, SKF_N_STATES, SKF_N_STATES, kH, eyeKalH);
    mTranspose(eyeKalH, SKF_N_STATES, SKF_N_STATES, eyeKalHT);
    mMultM(eyeKalH, SKF_N_STATES, SKF_N_STATES, covarBar, SKF_N_STATES, SKF_N_STATES, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, SKF_N_STATES, SKF_N_STATES, eyeKalHT, SKF_N_STATES, SKF_N_STATES, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, SKF_N_STATES, (size_t) numObs, noiseMat, (size_t) numObs, (size_t) numObs, kalR);
    mTranspose(kalmanGain, SKF_N_STATES, (size_t) numObs, kalT);
    mMultM(kalR, SKF_N_STATES, (size_t) numObs, kalT, (size_t) numObs, SKF_N_STATES, kalRKalT);
    mAdd(covar, SKF_N_STATES, SKF_N_STATES, kalRKalT, covar);
    
    
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
void sunlineEKFUpdate(double kalmanGain[SKF_N_STATES*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES*SKF_N_STATES], double qObsVal, int numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double *states, double *x, double *covar)
{

    double eye[SKF_N_STATES*SKF_N_STATES], kH[SKF_N_STATES*SKF_N_STATES];
    double eyeKalH[SKF_N_STATES*SKF_N_STATES], eyeKalHT[SKF_N_STATES*SKF_N_STATES];
    double eyeKalHCovarBar[SKF_N_STATES*SKF_N_STATES], kalR[SKF_N_STATES*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*SKF_N_STATES], kalRKalT[SKF_N_STATES*SKF_N_STATES];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Set variables to zero */
    mSetZero(kH, SKF_N_STATES, SKF_N_STATES);
    mSetZero(eyeKalH, SKF_N_STATES, SKF_N_STATES);
    mSetZero(eyeKalHT, SKF_N_STATES, SKF_N_STATES);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, SKF_N_STATES, SKF_N_STATES);
    mSetZero(kalRKalT, SKF_N_STATES, SKF_N_STATES);
    mSetZero(kalT, MAX_N_CSS_MEAS, SKF_N_STATES);
    mSetZero(kalR, SKF_N_STATES, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, SKF_N_STATES, SKF_N_STATES);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, (size_t) numObs, (size_t) numObs);
    mScale(qObsVal, noiseMat, (size_t) numObs, (size_t) numObs, noiseMat);
    
    /*! - Update the state error*/
    mMultV(kalmanGain, SKF_N_STATES, (size_t) numObs, yObs, x);

    /*! - Change the reference state*/
    vAdd(states, SKF_N_STATES, x, states);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, SKF_N_STATES, (size_t) numObs, hObs, (size_t) numObs, SKF_N_STATES, kH);
    mSetIdentity(eye, SKF_N_STATES, SKF_N_STATES);
    mSubtract(eye, SKF_N_STATES, SKF_N_STATES, kH, eyeKalH);
    mTranspose(eyeKalH, SKF_N_STATES, SKF_N_STATES, eyeKalHT);
    mMultM(eyeKalH, SKF_N_STATES, SKF_N_STATES, covarBar, SKF_N_STATES, SKF_N_STATES, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, SKF_N_STATES, SKF_N_STATES, eyeKalHT, SKF_N_STATES, SKF_N_STATES, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, SKF_N_STATES, (size_t) numObs, noiseMat, (size_t) numObs, (size_t) numObs, kalR);
    mTranspose(kalmanGain, SKF_N_STATES, (size_t) numObs, kalT);
    mMultM(kalR, SKF_N_STATES, (size_t) numObs, kalT, (size_t) numObs, SKF_N_STATES, kalRKalT);
    mAdd(covar, SKF_N_STATES, SKF_N_STATES, kalRKalT, covar);
    
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
 @param CBias Array of sensor biases
 @param obs Pointer to the observations
 @param yMeas Pointer to the innovation
 @param numObs Pointer to the number of observations
 @param measMat Point to the H measurement matrix
 */

void sunlineHMatrixYMeas(double states[SKF_N_STATES], int numCSS, double cssSensorCos[MAX_N_CSS_MEAS], double sensorUseThresh, double cssNHat_B[MAX_NUM_CSS_SENSORS*3], double CBias[MAX_NUM_CSS_SENSORS], double *obs, double *yMeas, int *numObs, double *measMat)
{
    int i, obsCounter;
    double sensorNormal[3];
    
    v3SetZero(sensorNormal);

    obsCounter = 0;
    /*! - Loop over all available coarse sun sensors and only use ones that meet validity threshold*/
    for(i=0; i<numCSS; i++)
    {
        if(cssSensorCos[i] > sensorUseThresh)
        {
            /*! - For each valid measurement, copy observation value and compute expected obs value and fill out H matrix.*/
            v3Scale(CBias[i], &(cssNHat_B[i*3]), sensorNormal); /* scaled sensor normal */

            *(obs+obsCounter) = cssSensorCos[i];
            *(yMeas+obsCounter) = cssSensorCos[i] - v3Dot(&(states[0]), sensorNormal);

            mSetSubMatrix(sensorNormal, 1, 3, measMat, MAX_NUM_CSS_SENSORS, SKF_N_STATES, obsCounter, 0);
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

void sunlineKalmanGain(double covarBar[SKF_N_STATES*SKF_N_STATES], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double qObsVal, int numObs, double *kalmanGain)
{
    double hObsT[SKF_N_STATES*MAX_N_CSS_MEAS];
    double covHT[SKF_N_STATES*MAX_N_CSS_MEAS];
    double hCovar[MAX_N_CSS_MEAS*SKF_N_STATES], hCovarHT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double rMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    
    /* Setting all local variables to zero */
    mSetZero(hObsT, SKF_N_STATES, MAX_N_CSS_MEAS);
    mSetZero(covHT, SKF_N_STATES, MAX_N_CSS_MEAS);
    mSetZero(hCovar, MAX_N_CSS_MEAS, SKF_N_STATES);
    mSetZero(hCovarHT, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(rMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    
    mTranspose(hObs, (size_t) numObs, SKF_N_STATES, hObsT);
    
    mMultM(covarBar, SKF_N_STATES, SKF_N_STATES, hObsT, SKF_N_STATES, (size_t) numObs, covHT);
    mMultM(hObs, (size_t) numObs, SKF_N_STATES, covarBar, SKF_N_STATES, SKF_N_STATES, hCovar);
    mMultM(hCovar, (size_t) numObs, SKF_N_STATES, hObsT, SKF_N_STATES, (size_t) numObs, hCovarHT);
    
    mSetIdentity(rMat, (size_t) (size_t) numObs, (size_t) numObs);
    mScale(qObsVal, rMat, (size_t) numObs, (size_t) numObs, rMat);
    
    /*! - Add measurement noise */
    mAdd(hCovarHT, (size_t) numObs, (size_t) numObs, rMat, hCovarHT);
    
    /*! - Invert the previous matrix */
    mInverse(hCovarHT, (size_t) numObs, hCovarHT);
    
    /*! - Compute the Kalman Gain */
    mMultM(covHT, SKF_N_STATES, (size_t) numObs, hCovarHT, (size_t) numObs, (size_t) numObs, kalmanGain);
    
}


