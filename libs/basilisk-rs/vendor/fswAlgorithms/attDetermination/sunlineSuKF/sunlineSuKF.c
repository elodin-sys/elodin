/*
 ISC License

 Copyright (c) 2016-2018, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "fswAlgorithms/attDetermination/sunlineSuKF/sunlineSuKF.h"
#include "architecture/utilities/ukfUtilities.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for theCSS WLS estimator.
 @return void
 @param configData The configuration data associated with the CSS WLS estimator
 @param moduleID The module identifier
 */
void SelfInit_sunlineSuKF(SunlineSuKFConfig *configData, int64_t moduleID)
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
void Reset_sunlineSuKF(SunlineSuKFConfig *configData, uint64_t callTime,
                      int64_t moduleID)
{
    
    CSSConfigMsgPayload cssConfigInBuffer;
    int32_t badUpdate;
    double tempMatrix[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];
    badUpdate = 0;

    /*! - Zero the local configuration data structures and outputs */
    mSetZero(configData->cssNHat_B, MAX_NUM_CSS_SENSORS, 3);
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();

    // check if the required input messages are included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineSuKF.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineSuKF.cssDataInMsg wasn't connected.");
    }

    /*! - Read in mass properties and coarse sun sensor configuration information.*/
    cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "sunlineSuKF.cssConfigInMsg.nCSS must not be greater than "
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
    configData->dt = 0.0;
    configData->timeTag = callTime*NANO2SEC;
    configData->numStates = SKF_N_STATES_SWITCH;
    configData->countHalfSPs = SKF_N_STATES_SWITCH;
    configData->numObs = MAX_N_CSS_MEAS;
    
    /*! Initalize the filter to use b_1 of the body frame to make frame*/
    v3Set(1, 0, 0, configData->bVec_B);
    configData->switchTresh = 0.866;
    
    /*! - Ensure that all internal filter matrices are zeroed*/
    vSetZero(configData->obs, configData->numObs);
    vSetZero(configData->wM, configData->countHalfSPs * 2 + 1);
    vSetZero(configData->wC, configData->countHalfSPs * 2 + 1);
    mSetZero(configData->sBar, configData->numStates, configData->numStates);
    mSetZero(configData->SP, configData->countHalfSPs * 2 + 1,
             configData->numStates);
    mSetZero(configData->sQnoise, configData->numStates, configData->numStates);
    
    /*! - Set lambda/gamma to standard value for unscented kalman filters */
    configData->lambdaVal = configData->alpha*configData->alpha*
    (configData->numStates + configData->kappa) - configData->numStates;
    configData->gamma = sqrt(configData->numStates + configData->lambdaVal);
    
    
    /*! - Set the wM/wC vectors to standard values for unscented kalman filters*/
    configData->wM[0] = configData->lambdaVal / (configData->numStates +
                                                 configData->lambdaVal);
    configData->wC[0] = configData->lambdaVal / (configData->numStates +
                                                 configData->lambdaVal) + (1 - configData->alpha*configData->alpha + configData->beta);
    for (uint32_t i = 1; i<configData->countHalfSPs * 2 + 1; i++)
    {
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates +
                                             configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    /*! - User a cholesky decomposition to obtain the sBar and sQnoise matrices for use in 
          filter at runtime*/
    mCopy(configData->covarInit, configData->numStates, configData->numStates,
          configData->covar);
    mSetZero(configData->covarPrev, configData->numStates, configData->numStates);
    mSetZero(configData->sBarPrev, configData->numStates, configData->numStates);
    vSetZero(configData->statePrev, configData->numStates);
    mCopy(configData->covar, configData->numStates, configData->numStates,
          configData->sBar);
    badUpdate += ukfCholDecomp(configData->sBar, (int32_t) configData->numStates,
                               (int32_t) configData->numStates, tempMatrix);
    mCopy(tempMatrix, configData->numStates, configData->numStates,
          configData->sBar);
    badUpdate += ukfCholDecomp(configData->qNoise, (int32_t)configData->numStates,
                               (int32_t) configData->numStates, configData->sQnoise);
    mTranspose(configData->sQnoise, configData->numStates,
               configData->numStates, configData->sQnoise);
    
    if (CSSArraySensorMsg_C_isWritten(&configData->cssDataInMsg)){
        configData->cssSensorInBuffer = CSSArraySensorMsg_C_read(&configData->cssDataInMsg);
    } else {
        configData->cssSensorInBuffer = CSSArraySensorMsg_C_zeroMsgPayload();
    }

    if (badUpdate <0){
        _bskLog(configData->bskLogger, BSK_WARNING, "Reset method contained bad update");
    }
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_sunlineSuKF(SunlineSuKFConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double newTimeTag;
    double yBar[MAX_N_CSS_MEAS];
    double tempYVec[MAX_N_CSS_MEAS];
    double sunheading_hat[3];
    double states_BN[SKF_N_STATES_SWITCH];
    uint64_t i;
    uint64_t timeOfMsgWritten;
    int isWritten;
    SunlineFilterMsgPayload sunlineDataOutBuffer;
    double maxSens;

    /*! - Read the input parsed CSS sensor data message*/
    configData->cssSensorInBuffer = CSSArraySensorMsg_C_read(&configData->cssDataInMsg);
    timeOfMsgWritten = CSSArraySensorMsg_C_timeWritten(&configData->cssDataInMsg);
    isWritten = CSSArraySensorMsg_C_isWritten(&configData->cssDataInMsg);

    /* zero the output messages */
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();
    sunlineDataOutBuffer = SunlineFilterMsg_C_zeroMsgPayload();

    /*! If the filter is not initialized manually, give it an initial guess using the CSS with the strongest signal.*/
    if(configData->filterInitialized==0)
    {
        vSetZero(configData->stateInit, SKF_N_STATES_SWITCH);
        configData->stateInit[5] = 1;
        configData->stateInit[0] = 1;
        maxSens = 0.0;
        /*! Loop through sensors to find max*/
        for(i=0; i<configData->numCSSTotal; i++)
        {
            if(configData->cssSensorInBuffer.CosValue[i] > maxSens)
            {
                v3Copy(&(configData->cssNHat_B[i*3]), configData->stateInit);
                maxSens = configData->cssSensorInBuffer.CosValue[i];
                /*! Max sensor reading is initial guess for the kelly factor*/
                configData->stateInit[5] = maxSens;
            }
        }
        if(maxSens < configData->sensorUseThresh)
        {
            return;
        }
        /*! The normal of the max activated sensor is the initial state*/
        vCopy(configData->stateInit, configData->numStates, configData->state);
        configData->filterInitialized = 1;
    }
    
    v3Normalize(&configData->state[0], sunheading_hat);
    
    /*! - Check for switching frames */
    if (fabs(v3Dot(configData->bVec_B, sunheading_hat)) > configData->switchTresh)
    {
        sunlineSuKFSwitch(configData->bVec_B, configData->state, configData->covar);
    }
    
    /*! - If the time tag from the measured data is new compared to previous step, 
          propagate and update the filter*/
    newTimeTag = timeOfMsgWritten * NANO2SEC;
    if(newTimeTag >= configData->timeTag && isWritten)
    {
        sunlineSuKFTimeUpdate(configData, newTimeTag);
        sunlineSuKFMeasUpdate(configData, newTimeTag);
    }
    v3Normalize(configData->state, configData->state);
    /*! - If current clock time is further ahead than the measured time, then
          propagate to this current time-step*/
    newTimeTag = callTime*NANO2SEC;
    if(newTimeTag > configData->timeTag)
    {
        sunlineSuKFTimeUpdate(configData, newTimeTag);
    }
    
    /*! - Compute Post Fit Residuals, first get Y (eq 22) using the states post fit*/
    sunlineSuKFMeasModel(configData);
    
    /*! - Compute the value for the yBar parameter (equation 23)*/
    vSetZero(yBar, configData->numObs);
    for(i=0; i<configData->countHalfSPs*2+1; i++)
    {
        vCopy(&(configData->yMeas[i*configData->numObs]), configData->numObs,
              tempYVec);
        vScale(configData->wM[i], tempYVec, configData->numObs, tempYVec);
        vAdd(yBar, configData->numObs, tempYVec, yBar);
    }
    
    /*! - The post fits are y- ybar*/
    mSubtract(configData->obs, configData->numObs, 1, yBar, configData->postFits);
    
    /*! - Write the sunline estimate into the copy of the navigation message structure*/
	v3Copy(configData->state, configData->outputSunline.vehSunPntBdy);
    v3Normalize(configData->outputSunline.vehSunPntBdy,
        configData->outputSunline.vehSunPntBdy);
    configData->outputSunline.timeTag = configData->timeTag;
    NavAttMsg_C_write(&configData->outputSunline, &configData->navStateOutMsg, moduleID, callTime);
    
    /*! - Switch the rates back to omega_BN instead of omega_SB */
    vCopy(configData->state, SKF_N_STATES_SWITCH, states_BN);
    vScale(-1, &(states_BN[3]), 2, &(states_BN[3])); /*! The Filter currently outputs omega_SB = -omega_BN (check doc for details)*/
    
    /*! - Populate the filter states output buffer and write the output message*/
    sunlineDataOutBuffer.timeTag = configData->timeTag;
    sunlineDataOutBuffer.numObs = (int) configData->numObs;
    memmove(sunlineDataOutBuffer.covar, configData->covar,
            SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH*sizeof(double));
    memmove(sunlineDataOutBuffer.state, states_BN, SKF_N_STATES_SWITCH*sizeof(double));
    memmove(sunlineDataOutBuffer.postFitRes, configData->postFits, MAX_N_CSS_MEAS*sizeof(double));
    SunlineFilterMsg_C_write(&sunlineDataOutBuffer, &configData->filtDataOutMsg, moduleID, callTime);
    
    return;
}

/*! This method propagates a sunline state vector forward in time.  Note 
    that the calling parameter is updated in place to save on data copies.
	@return void
    @param stateInOut The state that is propagated
    @param b_Vec b vector
    @param dt time step (s)
*/
void sunlineStateProp(double *stateInOut, double *b_Vec, double dt)
{

    double propagatedVel[SKF_N_STATES_HALF];
    double omegaCrossd[SKF_N_STATES_HALF];
    double omega_BN_S[SKF_N_STATES_HALF] = {0, -stateInOut[3], -stateInOut[4]};
    double omega_BN_B[SKF_N_STATES_HALF];
    double dcm_BS[SKF_N_STATES_HALF][SKF_N_STATES_HALF];

    mSetZero(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF);

    sunlineSuKFComputeDCM_BS(stateInOut, b_Vec, &dcm_BS[0][0]);
    mMultV(dcm_BS, SKF_N_STATES_HALF, SKF_N_STATES_HALF, omega_BN_S, omega_BN_B);
    /* Set local variables to zero*/
    vSetZero(propagatedVel, SKF_N_STATES_HALF);
    
    /*! Take omega cross d*/
    v3Cross(omega_BN_B, stateInOut, omegaCrossd);
    
    /*! - Multiply omega cross d by dt and add to state to propagate */
    v3Scale(-dt, omegaCrossd, propagatedVel);
    v3Add(stateInOut, propagatedVel, stateInOut);
    v3Normalize(stateInOut, stateInOut);
    
	return;
}

/*! This method performs the time update for the sunline kalman filter.
     It propagates the sigma points forward in time and then gets the current 
	 covariance and state estimates.
	 @return void
     @param configData The configuration data associated with the CSS estimator
     @param updateTime The time that we need to fix the filter to (seconds)
*/
int sunlineSuKFTimeUpdate(SunlineSuKFConfig *configData, double updateTime)
{
    int Index, badUpdate;
	double sBarT[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];
	double xComp[SKF_N_STATES_SWITCH], AT[(2 * SKF_N_STATES_SWITCH + SKF_N_STATES_SWITCH)*SKF_N_STATES_SWITCH];
	double aRow[SKF_N_STATES_SWITCH], rAT[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH], xErr[SKF_N_STATES_SWITCH]; 
	double sBarUp[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];
	double *spPtr;
    double procNoise[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];
    badUpdate = 0;
    
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    configData->dt = updateTime - configData->timeTag;
    mCopy(configData->sQnoise, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, procNoise);
    /*! - Copy over the current state estimate into the 0th Sigma point and propagate by dt*/
	vCopy(configData->state, configData->numStates,
		&(configData->SP[0 * configData->numStates + 0]));
	mSetZero(rAT, configData->countHalfSPs, configData->countHalfSPs);
	sunlineStateProp(&(configData->SP[0]), configData->bVec_B, configData->dt);
    /*! - Scale that Sigma point by the appopriate scaling factor (Wm[0])*/
	vScale(configData->wM[0], &(configData->SP[0]),
        configData->numStates, configData->xBar);
    /*! - Get the transpose of the sBar matrix because it is easier to extract Rows vs columns*/
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               sBarT);
    /*! - For each Sigma point, apply sBar-based error, propagate forward, and scale by Wm just like 0th.
          Note that we perform +/- sigma points simultaneously in loop to save loop values.*/
	for (uint64_t i = 0; i<configData->countHalfSPs; i++)
	{
		Index = (int) i + 1;
		spPtr = &(configData->SP[Index*(int)configData->numStates]);
		vCopy(&sBarT[i*(int)configData->numStates], configData->numStates, spPtr);
		vScale(configData->gamma, spPtr, configData->numStates, spPtr);
		vAdd(spPtr, configData->numStates, configData->state, spPtr);
		sunlineStateProp(spPtr, configData->bVec_B, configData->dt);
		vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
		vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
		
		Index = (int) i + 1 + (int) configData->countHalfSPs;
        spPtr = &(configData->SP[Index*(int)configData->numStates]);
        vCopy(&sBarT[i*(int) configData->numStates], configData->numStates, spPtr);
        vScale(-configData->gamma, spPtr, configData->numStates, spPtr);
        vAdd(spPtr, configData->numStates, configData->state, spPtr);
        sunlineStateProp(spPtr, configData->bVec_B, configData->dt);
        vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
        vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
	}
    /*! - Zero the AT matrix prior to assembly*/
    mSetZero(AT, (2 * configData->countHalfSPs + configData->numStates),
        configData->countHalfSPs);
	/*! - Assemble the AT matrix.  Note that this matrix is the internals of 
          the qr decomposition call in the source design documentation.  It is 
          the inside of equation 20 in that document*/
	for (uint64_t i = 0; i<2 * configData->countHalfSPs; i++)
	{
		
        vScale(-1.0, configData->xBar, configData->numStates, aRow);
        vAdd(aRow, configData->numStates,
             &(configData->SP[(i+1)*(int) configData->numStates]), aRow);
        if (configData->wC[i+1] <0){return -1;}
        vScale(sqrt(configData->wC[i+1]), aRow, configData->numStates, aRow);
		memcpy((void *)&AT[i*(int) configData->numStates], (void *)aRow,
			configData->numStates*sizeof(double));
	}
    
    /*! - Pop the sQNoise matrix on to the end of AT prior to getting QR decomposition*/
	memcpy(&AT[2 * configData->countHalfSPs*configData->numStates],
		procNoise, configData->numStates*configData->numStates
        *sizeof(double));
    /*! - QR decomposition (only R computed!) of the AT matrix provides the new sBar matrix*/
    ukfQRDJustR(AT, (int32_t) (2 * configData->countHalfSPs + configData->numStates),
                (int32_t) configData->countHalfSPs, rAT);
    mCopy(rAT, configData->numStates, configData->numStates, sBarT);
    mTranspose(sBarT, configData->numStates, configData->numStates,
        configData->sBar);
    
    /*! - Shift the sBar matrix over by the xBar vector using the appropriate weight 
          like in equation 21 in design document.*/
    vScale(-1.0, configData->xBar, configData->numStates, xErr);
    vAdd(xErr, configData->numStates, &configData->SP[0], xErr);
    badUpdate += ukfCholDownDate(configData->sBar, xErr, configData->wC[0],
                                 (int32_t) configData->numStates, sBarUp);
    
    /*! - Save current sBar matrix, covariance, and state estimate off for further use*/
    mCopy(sBarUp, configData->numStates, configData->numStates, configData->sBar);
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
        configData->covar);
	mMultM(configData->sBar, configData->numStates, configData->numStates,
        configData->covar, configData->numStates, configData->numStates,
           configData->covar);
    vCopy(&(configData->SP[0]), configData->numStates, configData->state );
	
	configData->timeTag = updateTime;
    
    if (badUpdate<0){
        sunlineSuKFCleanUpdate(configData);
        return -1;
    }
    return 0;
}

/*! This method computes what the expected measurement vector is for each CSS 
    that is present on the spacecraft.  All data is transacted from the main 
    data structure for the model because there are many variables that would 
    have to be updated otherwise.
 @return void
 @param configData The configuration data associated with the CSS estimator

 */
void sunlineSuKFMeasModel(SunlineSuKFConfig *configData)
{
    uint32_t i, j, obsCounter;
    double sensorNormal[3];
    double normalizedState[3];
    double stateNorm;
    double expectedMeas;
    double kellDelta;

    obsCounter = 0;
    /*! - Loop over all available coarse sun sensors and only use ones that meet validity threshold*/
    for(i=0; i<configData->numCSSTotal; i++)
    {
        v3Scale(configData->CBias[i], &(configData->cssNHat_B[i*3]), sensorNormal);
        stateNorm = v3Norm(configData->state);
        v3Normalize(configData->state, normalizedState);
        expectedMeas = v3Dot(normalizedState, sensorNormal);
        expectedMeas = expectedMeas > 0.0 ? expectedMeas : 0.0;
        kellDelta = 1.0;
        /*! - Scale the measurement by the kelly factor.*/
        if(configData->kellFits[i].cssKellFact > 0.0)
        {
            kellDelta -= exp(-pow(expectedMeas,configData->kellFits[i].cssKellPow) /
                             configData->kellFits[i].cssKellFact);
            expectedMeas *= kellDelta;
            expectedMeas *= configData->kellFits[i].cssRelScale;
        }
        expectedMeas *= configData->state[5];
        expectedMeas = expectedMeas > 0.0 ? expectedMeas : 0.0;
        if(configData->cssSensorInBuffer.CosValue[i] > configData->sensorUseThresh ||
           expectedMeas > configData->sensorUseThresh)
        {
            /*! - For each valid measurement, copy observation value and compute expected obs value
                  on a per sigma-point basis.*/
            configData->obs[obsCounter] = configData->cssSensorInBuffer.CosValue[i];
            for(j=0; j<configData->countHalfSPs*2+1; j++)
            {
                stateNorm = v3Norm(&(configData->SP[j*SKF_N_STATES_SWITCH]));
                v3Normalize(&(configData->SP[j*SKF_N_STATES_SWITCH]), normalizedState);
                expectedMeas = v3Dot(normalizedState, sensorNormal);
                expectedMeas = expectedMeas > 0.0 ? expectedMeas : 0.0;
                kellDelta = 1.0;
                /*! - Scale the measurement by the kelly factor.*/
                if(configData->kellFits[i].cssKellFact > 0.0)
                {
                    kellDelta -= exp(-pow(expectedMeas,configData->kellFits[i].cssKellPow) /
                                     configData->kellFits[i].cssKellFact);
                    expectedMeas *= kellDelta;
                    expectedMeas *= configData->kellFits[i].cssRelScale;
                }
                expectedMeas *= configData->SP[j*SKF_N_STATES_SWITCH+5];
                expectedMeas = expectedMeas > 0.0 ? expectedMeas : 0.0;
                configData->yMeas[obsCounter*(configData->countHalfSPs*2+1) + j] =
                    expectedMeas;
            }
            obsCounter++;
        }
    }
    /*! - yMeas matrix was set backwards deliberately so we need to transpose it through*/
    mTranspose(configData->yMeas, obsCounter, configData->countHalfSPs*2+1,
        configData->yMeas);
    configData->numObs = obsCounter;
    
}

/*! This method performs the measurement update for the sunline kalman filter.
 It applies the observations in the obs vectors to the current state estimate and 
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param updateTime The time that we need to fix the filter to (seconds)
 */
int sunlineSuKFMeasUpdate(SunlineSuKFConfig *configData, double updateTime)
{
    uint32_t i;
    int32_t badUpdate;
    double yBar[MAX_N_CSS_MEAS], syInv[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double kMat[SKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    double xHat[SKF_N_STATES_SWITCH], sBarT[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH], tempYVec[MAX_N_CSS_MEAS];
    double AT[(2 * SKF_N_STATES_SWITCH + MAX_N_CSS_MEAS)*MAX_N_CSS_MEAS], qChol[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double rAT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS], syT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double sy[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS], Ucol[SKF_N_STATES_SWITCH];
    double updMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS], pXY[SKF_N_STATES_SWITCH*MAX_N_CSS_MEAS], Umat[SKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];
    badUpdate = 0;
    
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    
    /*! - Compute the valid observations and the measurement model for all observations*/
    sunlineSuKFMeasModel(configData);
    
    /*! - Compute the value for the yBar parameter (note that this is equation 23 in the 
          time update section of the reference document*/
    vSetZero(yBar, configData->numObs);
    for(i=0; i<configData->countHalfSPs*2+1; i++)
    {
        vCopy(&(configData->yMeas[i*configData->numObs]), configData->numObs,
              tempYVec);
        vScale(configData->wM[i], tempYVec, configData->numObs, tempYVec);
        vAdd(yBar, configData->numObs, tempYVec, yBar);
    }
    
    /*! - Populate the matrix that we perform the QR decomposition on in the measurement 
          update section of the code.  This is based on the differenence between the yBar 
          parameter and the calculated measurement models.  Equation 24 in driving doc. */
    mSetZero(AT, configData->countHalfSPs*2+configData->numObs,
        configData->numObs);
    for(i=0; i<configData->countHalfSPs*2; i++)
    {
        vScale(-1.0, yBar, configData->numObs, tempYVec);
        vAdd(tempYVec, configData->numObs,
             &(configData->yMeas[(i+1)*configData->numObs]), tempYVec);
        if (configData->wC[i+1] <0){return -1;}
        vScale(sqrt(configData->wC[i+1]), tempYVec, configData->numObs, tempYVec);
        memcpy(&(AT[i*configData->numObs]), tempYVec,
               configData->numObs*sizeof(double));
    }
    
    /*! - This is the square-root of the Rk matrix which we treat as the Cholesky
        decomposition of the observation variance matrix constructed for our number 
        of observations*/
    mSetZero(configData->qObs, configData->numCSSTotal, configData->numCSSTotal);
    mSetIdentity(configData->qObs, configData->numObs, configData->numObs);
    mScale(configData->qObsVal, configData->qObs, configData->numObs,
           configData->numObs, configData->qObs);
    ukfCholDecomp(configData->qObs, (int32_t) configData->numObs, (int32_t) configData->numObs, qChol);
    memcpy(&(AT[2*configData->countHalfSPs*configData->numObs]),
           qChol, configData->numObs*configData->numObs*sizeof(double));
    /*! - Perform QR decomposition (only R again) of the above matrix to obtain the 
          current Sy matrix*/
    ukfQRDJustR(AT, (int32_t) (2*configData->countHalfSPs+configData->numObs),
                (int32_t) configData->numObs, rAT);
    mCopy(rAT, configData->numObs, configData->numObs, syT);
    mTranspose(syT, configData->numObs, configData->numObs, sy);
    /*! - Shift the matrix over by the difference between the 0th SP-based measurement 
          model and the yBar matrix (cholesky down-date again)*/
    vScale(-1.0, yBar, configData->numObs, tempYVec);
    vAdd(tempYVec, configData->numObs, &(configData->yMeas[0]), tempYVec);
    badUpdate += ukfCholDownDate(sy, tempYVec, configData->wC[0],
                                 (int32_t) configData->numObs, updMat);
    /*! - Shifted matrix represents the Sy matrix */
    mCopy(updMat, configData->numObs, configData->numObs, sy);
    mTranspose(sy, configData->numObs, configData->numObs, syT);

    /*! - Construct the Pxy matrix (equation 26) which multiplies the Sigma-point cloud 
          by the measurement model cloud (weighted) to get the total Pxy matrix*/
    mSetZero(pXY, configData->numStates, configData->numObs);
    for(i=0; i<2*configData->countHalfSPs+1; i++)
    {
        vScale(-1.0, yBar, configData->numObs, tempYVec);
        vAdd(tempYVec, configData->numObs,
             &(configData->yMeas[i*configData->numObs]), tempYVec);
        vSubtract(&(configData->SP[i*configData->numStates]), configData->numStates,
                  configData->xBar, xHat);
        vScale(configData->wC[i], xHat, configData->numStates, xHat);
        mMultM(xHat, configData->numStates, 1, tempYVec, 1, configData->numObs,
            kMat);
        mAdd(pXY, configData->numStates, configData->numObs, kMat, pXY);
    }

    /*! - Then we need to invert the SyT*Sy matrix to get the Kalman gain factor.  Since
          The Sy matrix is lower triangular, we can do a back-sub inversion instead of 
          a full matrix inversion.  That is the ukfUInv and ukfLInv calls below.  Once that 
          multiplication is done (equation 27), we have the Kalman Gain.*/
    badUpdate += ukfUInv(syT, (int32_t) configData->numObs, (int32_t) configData->numObs, syInv);
    
    mMultM(pXY, configData->numStates, configData->numObs, syInv,
           configData->numObs, configData->numObs, kMat);
    badUpdate += ukfLInv(sy, (int32_t) configData->numObs, (int32_t) configData->numObs, syInv);
    mMultM(kMat, configData->numStates, configData->numObs, syInv,
           configData->numObs, configData->numObs, kMat);
    
    
    /*! - Difference the yBar and the observations to get the observed error and 
          multiply by the Kalman Gain to get the state update.  Add the state update 
          to the state to get the updated state value (equation 27).*/
    vSubtract(configData->obs, configData->numObs, yBar, tempYVec);
    mMultM(kMat, configData->numStates, configData->numObs, tempYVec,
        configData->numObs, 1, xHat);
    vAdd(configData->state, configData->numStates, xHat, configData->state);
    /*! - Compute the updated matrix U from equation 28.  Note that I then transpose it 
         so that I can extract "columns" from adjacent memory*/
    mMultM(kMat, configData->numStates, configData->numObs, sy,
           configData->numObs, configData->numObs, Umat);
    mTranspose(Umat, configData->numStates, configData->numObs, Umat);
    /*! - For each column in the update matrix, perform a cholesky down-date on it to
     get the total shifted S matrix (called sBar in internal parameters*/
    for(i=0; i<configData->numObs; i++)
    {
        vCopy(&(Umat[i*configData->numStates]), configData->numStates, Ucol);
        badUpdate += ukfCholDownDate(configData->sBar, Ucol, -1.0, (int32_t) configData->numStates, sBarT);
        mCopy(sBarT, configData->numStates, configData->numStates,
              configData->sBar);
    }
    /*! - Compute equivalent covariance based on updated sBar matrix*/
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               configData->covar);
    mMultM(configData->sBar, configData->numStates, configData->numStates,
           configData->covar, configData->numStates, configData->numStates,
           configData->covar);
    
    if (badUpdate<0){
        sunlineSuKFCleanUpdate(configData);
        return -1;
    }
    return 0;
}


/*! This method computes the dcms necessary for the switch between the two frames.
 It the switches the states and the covariance, and sets s2 to be the new, different vector of the body frame.
 @return void
 @param bVec_B Pointer to b vector
 @param states Pointer to the states
 @param covar Pointer to the covariance
 */

void sunlineSuKFSwitch(double *bVec_B, double *states, double *covar)
{
    double dcm_BSold[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double dcm_BSnew_T[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double dcm_SnewSold[SKF_N_STATES_HALF][SKF_N_STATES_HALF];
    double switchMatP[SKF_N_STATES_SWITCH][SKF_N_STATES_SWITCH];
    double switchMat[SKF_N_STATES_SWITCH][SKF_N_STATES_SWITCH];
    
    double sun_heading_norm[SKF_N_STATES_HALF];
    double b1[SKF_N_STATES_HALF];
    double b2[SKF_N_STATES_HALF];
    
    /*!  Set the body frame vectors*/
    v3Set(1, 0, 0, b1);
    v3Set(0, 1, 0, b2);
    v3Normalize(&(states[0]), sun_heading_norm);
    
    /*! Populate the dcm_BS with the "old" S-frame*/
    sunlineSuKFComputeDCM_BS(sun_heading_norm, bVec_B, &dcm_BSold[0][0]);
    
    if (v3IsEqual(bVec_B, b1, 1e-10))
    {
        sunlineSuKFComputeDCM_BS(sun_heading_norm, b2, &dcm_BSnew_T[0][0]);
        v3Copy(b2, bVec_B);
    }
    else
    {
        sunlineSuKFComputeDCM_BS(sun_heading_norm, b1, &dcm_BSnew_T[0][0]);
        v3Copy(b1, bVec_B);
    }
    
    mTranspose(dcm_BSnew_T, SKF_N_STATES_HALF, SKF_N_STATES_HALF, dcm_BSnew_T);
    mMultM(dcm_BSnew_T, 3, 3, dcm_BSold, 3, 3, dcm_SnewSold);
    
    mSetIdentity(switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH);
    mSetSubMatrix(&dcm_SnewSold[1][1], 1, 2, &switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, 3, 3);
    mSetSubMatrix(&dcm_SnewSold[2][1], 1, 2, &switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, 4, 3);
    
    mMultV(switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, states, states);
    mMultM(switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, covar, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, switchMatP);
    mTranspose(switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, switchMat);
    mMultM(switchMatP, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, switchMat, SKF_N_STATES_SWITCH, SKF_N_STATES_SWITCH, covar);
    return;
}


void sunlineSuKFComputeDCM_BS(double sunheading[SKF_N_STATES_HALF], double bVec[SKF_N_STATES_HALF], double *dcm){
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

/*! This method cleans the filter states after a bad upadate on the fly.
 It removes the potentially corrupted previous estimates and puts the filter
 back to a working state.
 @return void
 @param configData The configuration data associated with the CSS estimator
 */
void sunlineSuKFCleanUpdate(SunlineSuKFConfig *configData){
    int i;
    /*! - Reset the observations, state, and covariannces to a previous safe value*/
    vSetZero(configData->obs, configData->numObs);
    vCopy(configData->statePrev, configData->numStates, configData->state);
    mCopy(configData->sBarPrev, configData->numStates, configData->numStates, configData->sBar);
    mCopy(configData->covarPrev, configData->numStates, configData->numStates, configData->covar);
    
    /*! - Reset the wM/wC vectors to standard values for unscented kalman filters*/
    configData->wM[0] = configData->lambdaVal / (configData->numStates +
                                                 configData->lambdaVal);
    configData->wC[0] = configData->lambdaVal / (configData->numStates +
                                                 configData->lambdaVal) + (1 - configData->alpha*configData->alpha + configData->beta);
    for (i = 1; i< ((int)configData->countHalfSPs) * 2 + 1; i++)
    {
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates +
                                             configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    return;
}
