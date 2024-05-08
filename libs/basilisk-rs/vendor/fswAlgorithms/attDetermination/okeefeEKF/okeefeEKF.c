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

#include "fswAlgorithms/attDetermination/okeefeEKF/okeefeEKF.h"
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
void SelfInit_okeefeEKF(okeefeEKFConfig *configData, int64_t moduleID)
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
void Reset_okeefeEKF(okeefeEKFConfig *configData, uint64_t callTime,
                      int64_t moduleID)
{
    
    CSSConfigMsgPayload cssConfigInBuffer;

    /*! - Zero the local configuration data structures and outputs */
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();
    mSetZero(configData->cssNHat_B, MAX_NUM_CSS_SENSORS, 3);

    // check if the required input messages are included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: okeefeEKF.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: okeefeEKF.cssDataInMsg wasn't connected.");
    }

    /*! - Read in coarse sun sensor configuration information.*/
    cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "okeefeEKF.cssConfigInMsg.nCSS must not be greater than "
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
    configData->numStates = SKF_N_STATES_HALF;
    configData->numObs = MAX_N_CSS_MEAS;
    
    /*! - Ensure that all internal filter matrices are zeroed*/
    vSetZero(configData->obs, configData->numObs);
    vSetZero(configData->yMeas, configData->numObs);
    vSetZero(configData->xBar, configData->numStates);
//    vSetZero(configData->omega, configData->numStates);
    vSetZero(configData->prev_states, configData->numStates);
    
    mSetZero(configData->covarBar, configData->numStates, configData->numStates);
    mSetZero(configData->dynMat, configData->numStates, configData->numStates);
    mSetZero(configData->measMat, configData->numObs, configData->numStates);
    mSetZero(configData->kalmanGain, configData->numStates, configData->numObs);
    mSetZero(configData->measNoise, configData->numObs, configData->numObs);
    
    mSetIdentity(configData->stateTransition, configData->numStates, configData->numStates);
    mSetIdentity(configData->procNoise,  configData->numStates, configData->numStates);
    mScale(configData->qProcVal, configData->procNoise, configData->numStates, configData->numStates, configData->procNoise);
    
    return;
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_okeefeEKF(okeefeEKFConfig *configData, uint64_t callTime,
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
        vCopy(configData->xBar, SKF_N_STATES_HALF, configData->x);
        mCopy(configData->covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, configData->covar);
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
    sunlineDataOutBuffer.numObs = (int) configData->numObs;
    memmove(sunlineDataOutBuffer.covar, configData->covar,
            SKF_N_STATES_HALF*SKF_N_STATES_HALF*sizeof(double));
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
void sunlineTimeUpdate(okeefeEKFConfig *configData, double updateTime)
{
    double stmT[SKF_N_STATES_HALF*SKF_N_STATES_HALF], covPhiT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double qGammaT[SKF_N_STATES_HALF*SKF_N_STATES_HALF], gammaQGammaT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    
	/*! Compute time step */
	configData->dt = updateTime - configData->timeTag;
    
    /*! - Propagate the previous reference states and STM to the current time */
    sunlineDynMatrixOkeefe(configData->omega, configData->dt, configData->dynMat);
    sunlineStateSTMProp(configData->dynMat, configData->dt, configData->omega, configData->state, configData->prev_states, configData->stateTransition);
    sunlineRateCompute(configData->state, configData->dt, configData->prev_states, configData->omega);


    /* xbar = Phi*x */
    mMultV(configData->stateTransition, SKF_N_STATES_HALF, SKF_N_STATES_HALF, configData->x, configData->xBar);
    
    /*! - Update the covariance */
    /*Pbar = Phi*P*Phi^T + Gamma*Q*Gamma^T*/
    mTranspose(configData->stateTransition, SKF_N_STATES_HALF, SKF_N_STATES_HALF, stmT);
    mMultM(configData->covar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, stmT, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covPhiT);
    mMultM(configData->stateTransition, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covPhiT, SKF_N_STATES_HALF, SKF_N_STATES_HALF, configData->covarBar);
    
    /*Compute Gamma and add gammaQGamma^T to Pbar. This is the process noise addition*/
    double Gamma[3][3]={{configData->dt*configData->dt/2,0,0},{0,configData->dt*configData->dt/2,0},{0,0,configData->dt*configData->dt/2}};
    
    mMultMt(configData->procNoise, SKF_N_STATES_HALF, SKF_N_STATES_HALF, Gamma, SKF_N_STATES_HALF, SKF_N_STATES_HALF, qGammaT);
    mMultM(Gamma, SKF_N_STATES_HALF, SKF_N_STATES_HALF, qGammaT, SKF_N_STATES_HALF, SKF_N_STATES_HALF, gammaQGammaT);
    mAdd(configData->covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, gammaQGammaT, configData->covarBar);
    
	configData->timeTag = updateTime;
}


/*! This method computes the rotation rate of the spacecraft by using the two previous state estimates.
	@return void
    @param states Updated states
    @param dt Time step
    @param prev_states The states saved from previous step for this purpose
    @param omega Pointer to the rotation rate
 */
void sunlineRateCompute(double states[SKF_N_STATES_HALF], double dt, double prev_states[SKF_N_STATES_HALF], double *omega)
{
    
    double dk_dot_dkmin1, dk_dot_dkmin1_normal, dk_cross_dkmin1_normal[SKF_N_STATES_HALF];
    double dk_hat[SKF_N_STATES_HALF], dkmin1_hat[SKF_N_STATES_HALF];

    if (dt < 1E-10){
    v3SetZero(omega);
    }
    
    else{
        if (v3IsZero(prev_states, 1E-10)){
        
            v3SetZero(omega);
        }
        else{
            /* Set local variables to zero */
            dk_dot_dkmin1=0;
            dk_dot_dkmin1_normal=0;
            vSetZero(dk_hat, SKF_N_STATES_HALF);
            vSetZero(dkmin1_hat, SKF_N_STATES_HALF);
        
            /* Normalized d_k and d_k-1 */
            v3Normalize(states, dk_hat);
            v3Normalize(prev_states, dkmin1_hat);
            
            /* Get the dot product to use in acos computation*/
            dk_dot_dkmin1_normal = v3Dot(dk_hat, dkmin1_hat);
            
            /*Get the cross prodcut for the direction of omega*/
            v3Cross(dk_hat, dkmin1_hat, dk_cross_dkmin1_normal);
            
            /* Scale direction by the acos and the 1/dt, and robustly compute arcos of angle*/
            if(dk_dot_dkmin1_normal>1){
                v3Scale(1/dt*safeAcos(1), dk_cross_dkmin1_normal, omega);
            }
            else if(dk_dot_dkmin1_normal<-1){
                v3Scale(1/dt*safeAcos(-1), dk_cross_dkmin1_normal, omega);
            }
            else {
                v3Scale(1/dt*safeAcos(dk_dot_dkmin1_normal), dk_cross_dkmin1_normal, omega);
            }
        }
    }
    return;
}


/*! @brief This method propagates a sunline state vector forward in time.  Note that the calling parameter is updated in place to save on data copies. This also updates the STM using the dynamics matrix.
	@return void
    @param dynMat dynamic matrix
    @param dt time step
    @param omega angular velocity
    @param stateInOut pointer to a state array
    @param prevstates pointer to previous states
    @param stateTransition pointer to state transition matrix
 */
void sunlineStateSTMProp(double dynMat[SKF_N_STATES_HALF*SKF_N_STATES_HALF], double dt, double omega[SKF_N_STATES_HALF], double *stateInOut, double *prevstates, double *stateTransition)
{
    
    double propagatedVel[SKF_N_STATES_HALF];
    double omegaCrossd[SKF_N_STATES_HALF];
    double deltatASTM[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    
    /* Populate d_k-1 */
    vCopy(stateInOut, SKF_N_STATES_HALF, prevstates);
    
    /* Set local variables to zero*/
    mSetZero(deltatASTM, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    vSetZero(propagatedVel, SKF_N_STATES_HALF);
    
    /*! Begin state update steps */
    /*! Take omega cross d*/
    v3Cross(omega, stateInOut, omegaCrossd);

    /*! - Multiply omega cross d by -dt and add to state to propagate */
    v3Scale(-dt, omegaCrossd, propagatedVel);
    v3Add(stateInOut, propagatedVel, stateInOut);
    
    /*! Begin STM propagation step */
    mSetIdentity(stateTransition, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mScale(dt, dynMat, SKF_N_STATES_HALF, SKF_N_STATES_HALF, deltatASTM);
    mAdd(stateTransition, SKF_N_STATES_HALF, SKF_N_STATES_HALF, deltatASTM, stateTransition);
    
    return;
}

/*! This method computes the dynamics matrix, which is the derivative of the
 dynamics F by the state X, evaluated at the reference state. It takes in the
 configure data and updates this A matrix pointer called dynMat
 @return void
 @param omega The rotation rate
 @param dt Time step
 @param dynMat Pointer to the Dynamic Matrix
 */

void sunlineDynMatrixOkeefe(double omega[SKF_N_STATES_HALF], double dt, double *dynMat)
{
    double skewOmega[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double negskewOmega[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    
    v3Tilde(omega, skewOmega);
    m33Scale(-1, skewOmega, negskewOmega);
    mCopy(negskewOmega, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dynMat);
    
    return;
}


/*! This method performs the measurement update for the sunline kalman filter.
 It applies the observations in the obs vectors to the current state estimate and
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param updateTime The time that we need to fix the filter to (seconds)
 */
void sunlineMeasUpdate(okeefeEKFConfig *configData, double updateTime)
{
    /*! - Compute the valid observations and the measurement model for all observations*/
    int numObsInt = (int) configData->numObs;
    sunlineHMatrixYMeas(configData->state, configData->numCSSTotal, configData->cssSensorInBuffer.CosValue, configData->sensorUseThresh, configData->cssNHat_B,
                        configData->CBias, configData->obs, configData->yMeas, &(numObsInt), configData->measMat);
    configData->numObs = (size_t) numObsInt;
    
    /*! - Compute the Kalman Gain. */
    sunlineKalmanGainOkeefe(configData->covarBar, configData->measMat, configData->qObsVal, (int) configData->numObs, configData->kalmanGain);
    
    /* Logic to switch from EKF to CKF. If the covariance is too large, switching references through an EKF could lead to filter divergence in extreme cases. In order to remedy this, past a certain infinite norm of the covariance, we update with a CKF in order to bring down the covariance. */
    
    if (vMaxAbs(configData->covar, SKF_N_STATES_HALF*SKF_N_STATES_HALF) > configData->eKFSwitch){
    /*! - Compute the update with a CKF */
    sunlineCKFUpdateOkeefe(configData->xBar, configData->kalmanGain, configData->covarBar, configData->qObsVal, (int) configData->numObs, configData->yMeas, configData->measMat, configData->x,configData->covar);
    }
    else{
    /*! - Compute the update with a EKF, notice the reference state is added as an argument because it is changed by the filter update */
    okeefeEKFUpdate(configData->kalmanGain, configData->covarBar, configData->qObsVal, (int) configData->numObs, configData->yMeas, configData->measMat, configData->state, configData->x, configData->covar);
    }
}

/*! This method computes the updated with a Classical Kalman Filter
 @return void
 @param xBar The state after a time update
 @param kalmanGain The computed Kalman Gain
 @param covarBar The time updated covariance
 @param qObsVal The observation noise
 @param numObsInt The amount of CSSs that get measurements
 @param yObs The y vector after receiving the measurements
 @param hObs The H matrix filled with the observations
 @param x Pointer to the state error for modification
 @param covar Pointer to the covariance after update
 */

void sunlineCKFUpdateOkeefe(double xBar[SKF_N_STATES_HALF], double kalmanGain[SKF_N_STATES_HALF*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES_HALF*SKF_N_STATES_HALF], double qObsVal, int numObsInt, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], double *x, double *covar)
{
    double measMatx[MAX_N_CSS_MEAS], innov[MAX_N_CSS_MEAS], kInnov[SKF_N_STATES_HALF];
    double eye[SKF_N_STATES_HALF*SKF_N_STATES_HALF], kH[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double eyeKalH[SKF_N_STATES_HALF*SKF_N_STATES_HALF], eyeKalHT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double eyeKalHCovarBar[SKF_N_STATES_HALF*SKF_N_STATES_HALF], kalR[SKF_N_STATES_HALF*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], kalRKalT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    size_t numObs = (size_t) numObsInt;

    /* Set variables to zero */
    mSetZero(kH, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(eyeKalHT, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(kalRKalT, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(kalT, MAX_N_CSS_MEAS, SKF_N_STATES_HALF);
    mSetZero(kalR, SKF_N_STATES_HALF, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, numObs, numObs);
    mScale(qObsVal, noiseMat, numObs, numObs, noiseMat);
    
    /*! - Compute innovation, multiply it my Kalman Gain, and add it to xBar*/
    mMultM(hObs, numObs, SKF_N_STATES_HALF, xBar, SKF_N_STATES_HALF, 1, measMatx);
    vSubtract(yObs, numObs, measMatx, innov);
    mMultM(kalmanGain, SKF_N_STATES_HALF, numObs, innov, numObs, 1, kInnov);
    vAdd(xBar, SKF_N_STATES_HALF, kInnov, x);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, SKF_N_STATES_HALF, numObs, hObs, numObs, SKF_N_STATES_HALF, kH);
    mSetIdentity(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSubtract(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF, kH, eyeKalH);
    mTranspose(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHT);
    mMultM(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHT, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, SKF_N_STATES_HALF, numObs, noiseMat, numObs, numObs, kalR);
    mTranspose(kalmanGain, SKF_N_STATES_HALF, numObs, kalT);
    mMultM(kalR, SKF_N_STATES_HALF, numObs, kalT, numObs, SKF_N_STATES_HALF, kalRKalT);
    mAdd(covar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, kalRKalT, covar);
    
    
}

/*! This method computes the updated with a Extended Kalman Filter
 @return void
 @param kalmanGain The computed Kalman Gain
 @param covarBar The time updated covariance
 @param qObsVal The observation noise
 @param numObsInt The amount of CSSs that get measurements
 @param yObs The y vector after receiving the measurements
 @param hObs The H matrix filled with the observations
 @param states Pointer to the states
 @param x Pointer to the state error for modification
 @param covar Pointer to the covariance after update
 */
void okeefeEKFUpdate(double kalmanGain[SKF_N_STATES_HALF*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES_HALF*SKF_N_STATES_HALF], double qObsVal, int numObsInt, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], double *states, double *x, double *covar)
{

    double eye[SKF_N_STATES_HALF*SKF_N_STATES_HALF], kH[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double eyeKalH[SKF_N_STATES_HALF*SKF_N_STATES_HALF], eyeKalHT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double eyeKalHCovarBar[SKF_N_STATES_HALF*SKF_N_STATES_HALF], kalR[SKF_N_STATES_HALF*MAX_N_CSS_MEAS];
    double kalT[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], kalRKalT[SKF_N_STATES_HALF*SKF_N_STATES_HALF];
    double noiseMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    size_t numObs = (size_t) numObsInt;

    /* Set variables to zero */
    mSetZero(kH, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(eyeKalHT, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(noiseMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(kalRKalT, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSetZero(kalT, MAX_N_CSS_MEAS, SKF_N_STATES_HALF);
    mSetZero(kalR, SKF_N_STATES_HALF, MAX_N_CSS_MEAS);
    mSetZero(eyeKalHCovarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    
    /* Set noise matrix given number of observations */
    mSetIdentity(noiseMat, numObs, numObs);
    mScale(qObsVal, noiseMat, numObs, numObs, noiseMat);
    
    /*! - Update the state error*/
    mMultV(kalmanGain, SKF_N_STATES_HALF, numObs, yObs, x);

    /*! - Change the reference state*/
    vAdd(states, SKF_N_STATES_HALF, x, states);
    
    /*! - Compute new covariance with Joseph's method*/
    mMultM(kalmanGain, SKF_N_STATES_HALF, numObs, hObs, numObs, SKF_N_STATES_HALF, kH);
    mSetIdentity(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF);
    mSubtract(eye, SKF_N_STATES_HALF, SKF_N_STATES_HALF, kH, eyeKalH);
    mTranspose(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHT);
    mMultM(eyeKalH, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHCovarBar);
    mMultM(eyeKalHCovarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, eyeKalHT, SKF_N_STATES_HALF, SKF_N_STATES_HALF, covar);
    
    /* Add noise to the covariance*/
    mMultM(kalmanGain, SKF_N_STATES_HALF, numObs, noiseMat, (size_t) numObs, (size_t) numObs, kalR);
    mTranspose(kalmanGain, SKF_N_STATES_HALF, (size_t) numObs, kalT);
    mMultM(kalR, SKF_N_STATES_HALF, (size_t) numObs, kalT, (size_t) numObs, SKF_N_STATES_HALF, kalRKalT);
    mAdd(covar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, kalRKalT, covar);
    
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
 @param CBias Vector of biases
 */

void sunlineHMatrixYMeas(double states[SKF_N_STATES_HALF], size_t numCSS, double cssSensorCos[MAX_N_CSS_MEAS], double sensorUseThresh, double cssNHat_B[MAX_NUM_CSS_SENSORS*3], double CBias[MAX_NUM_CSS_SENSORS], double *obs, double *yMeas, int *numObs, double *measMat)
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
            v3Scale(CBias[i], &(cssNHat_B[i*3]), sensorNormal); /* scaled sensor normal */
            
            *(obs+obsCounter) = cssSensorCos[i];
            *(yMeas+obsCounter) = cssSensorCos[i] - v3Dot(&(states[0]), sensorNormal);
            mSetSubMatrix(sensorNormal, 1, 3, measMat, MAX_NUM_CSS_SENSORS, SKF_N_STATES_HALF, obsCounter, 0);
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
 @param numObsInt The number of observations
 @param kalmanGain Pointer to the Kalman Gain
 */

void sunlineKalmanGainOkeefe(double covarBar[SKF_N_STATES_HALF*SKF_N_STATES_HALF], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], double qObsVal, int numObsInt, double *kalmanGain)
{
    double hObsT[SKF_N_STATES_HALF*MAX_N_CSS_MEAS];
    double covHT[SKF_N_STATES_HALF*MAX_N_CSS_MEAS];
    double hCovar[MAX_N_CSS_MEAS*SKF_N_STATES_HALF], hCovarHT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double rMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    size_t numObs;
    numObs = (size_t) numObsInt;
    
    /* Setting all local variables to zero */
    mSetZero(hObsT, SKF_N_STATES_HALF, MAX_N_CSS_MEAS);
    mSetZero(covHT, SKF_N_STATES_HALF, MAX_N_CSS_MEAS);
    mSetZero(hCovar, MAX_N_CSS_MEAS, SKF_N_STATES_HALF);
    mSetZero(hCovarHT, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    mSetZero(rMat, MAX_N_CSS_MEAS, MAX_N_CSS_MEAS);
    
    mTranspose(hObs, (size_t) numObs, SKF_N_STATES_HALF, hObsT);
    
    mMultM(covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, hObsT, SKF_N_STATES_HALF, (size_t) numObs, covHT);
    mMultM(hObs, (size_t) numObs, SKF_N_STATES_HALF, covarBar, SKF_N_STATES_HALF, SKF_N_STATES_HALF, hCovar);
    mMultM(hCovar, (size_t) numObs, SKF_N_STATES_HALF, hObsT, SKF_N_STATES_HALF, (size_t) numObs, hCovarHT);
    
    mSetIdentity(rMat, (size_t) numObs, (size_t) numObs);
    mScale(qObsVal, rMat, (size_t) numObs, (size_t) numObs, rMat);
    
    /*! - Add measurement noise */
    mAdd(hCovarHT, (size_t) numObs, (size_t) numObs, rMat, hCovarHT);
    
    /*! - Invert the previous matrix */
    mInverse(hCovarHT, (size_t) numObs, hCovarHT);
    
    /*! - Compute the Kalman Gain */
    mMultM(covHT, SKF_N_STATES_HALF, numObs, hCovarHT, numObs, numObs, kalmanGain);
    
}


