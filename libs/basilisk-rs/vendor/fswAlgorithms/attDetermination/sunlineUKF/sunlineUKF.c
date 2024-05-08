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

#include "fswAlgorithms/attDetermination/sunlineUKF/sunlineUKF.h"
#include "architecture/utilities/ukfUtilities.h"
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
void SelfInit_sunlineUKF(SunlineUKFConfig *configData, int64_t moduleID)
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
void Reset_sunlineUKF(SunlineUKFConfig *configData, uint64_t callTime,
                      int64_t moduleID)
{
    
    CSSConfigMsgPayload cssConfigInBuffer;
    double tempMatrix[SKF_N_STATES*SKF_N_STATES];
    
    /*! - Zero the local configuration data structures and outputs */
    configData->outputSunline = NavAttMsg_C_zeroMsgPayload();
    mSetZero(configData->cssNHat_B, MAX_NUM_CSS_SENSORS, 3);

    // check if the required input messages are included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineUKF.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: sunlineUKF.cssDataInMsg wasn't connected.");
    }

    /*! - Read in mass properties and coarse sun sensor configuration information.*/
    cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "sunlineUKF.cssConfigInMsg.nCSS must not be greater than "
                                                  "MAX_N_CSS_MEAS value.");
    }

    /*! - For each coarse sun sensor, convert the configuration data over from structure to body*/
    for(uint32_t i=0; i<cssConfigInBuffer.nCSS; i = i+1)
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
    configData->countHalfSPs = SKF_N_STATES;
    configData->numObs = MAX_N_CSS_MEAS;
    
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
    for (int i = 1; i<configData->countHalfSPs * 2 + 1; i++)
    {
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates +
                                             configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    /*! - User a cholesky decomposition to obtain the sBar and sQnoise matrices for use in 
          filter at runtime*/
    mCopy(configData->covar, configData->numStates, configData->numStates,
          configData->sBar);
    ukfCholDecomp(configData->sBar, configData->numStates,
                  configData->numStates, tempMatrix);
    mCopy(tempMatrix, configData->numStates, configData->numStates,
          configData->sBar);
    ukfCholDecomp(configData->qNoise, configData->numStates,
                  configData->numStates, configData->sQnoise);
    mTranspose(configData->sQnoise, configData->numStates,
               configData->numStates, configData->sQnoise);
    

    return;
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_sunlineUKF(SunlineUKFConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double newTimeTag;
    double yBar[MAX_N_CSS_MEAS];
    double tempYVec[MAX_N_CSS_MEAS];
    int i;
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
        sunlineUKFTimeUpdate(configData, newTimeTag);
        sunlineUKFMeasUpdate(configData, newTimeTag);
    }
    
    /*! - If current clock time is further ahead than the measured time, then
          propagate to this current time-step*/
    newTimeTag = callTime*NANO2SEC;
    if(newTimeTag > configData->timeTag)
    {
        sunlineUKFTimeUpdate(configData, newTimeTag);
    }
    
    /*! - Compute Post Fit Residuals, first get Y (eq 22) using the states post fit*/
    sunlineUKFMeasModel(configData);
    
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
    mSubtract(configData->obs, MAX_N_CSS_MEAS, 1, yBar, configData->postFits);
    
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
    memmove(sunlineDataOutBuffer.postFitRes, configData->postFits, MAX_N_CSS_MEAS*sizeof(double));
    SunlineFilterMsg_C_write(&sunlineDataOutBuffer, &configData->filtDataOutMsg, moduleID, callTime);

    return;
}

/*! This method propagates a sunline state vector forward in time.  Note 
    that the calling parameter is updated in place to save on data copies.
	@return void
    @param stateInOut The state that is propagated
    @param dt Time step (s)
*/
void sunlineStateProp(double *stateInOut, double dt)
{

    double propagatedVel[3];
    double pointUnit[3];
    double unitComp;
    
    /*! - Unitize the current estimate to find direction to restrict motion*/
    v3Normalize(stateInOut, pointUnit);
    unitComp = v3Dot(&(stateInOut[3]), pointUnit);
    v3Scale(unitComp, pointUnit, pointUnit);
    /*! - Subtract out rotation in the sunline axis because that is not observable 
          for coarse sun sensors*/
    v3Subtract(&(stateInOut[3]), pointUnit, &(stateInOut[3]));
    v3Scale(dt, &(stateInOut[3]), propagatedVel);
    v3Add(stateInOut, propagatedVel, stateInOut);

	return;
}

/*! This method performs the time update for the sunline kalman filter.
     It propagates the sigma points forward in time and then gets the current 
	 covariance and state estimates.
	 @return void
     @param configData The configuration data associated with the CSS estimator
     @param updateTime The time that we need to fix the filter to (seconds)
*/
void sunlineUKFTimeUpdate(SunlineUKFConfig *configData, double updateTime)
{
	int i, Index;
	double sBarT[SKF_N_STATES*SKF_N_STATES];
	double xComp[SKF_N_STATES], AT[(2 * SKF_N_STATES + SKF_N_STATES)*SKF_N_STATES];
	double aRow[SKF_N_STATES], rAT[SKF_N_STATES*SKF_N_STATES], xErr[SKF_N_STATES]; 
	double sBarUp[SKF_N_STATES*SKF_N_STATES];
	double *spPtr;

    /*! Compute time step */
	configData->dt = updateTime - configData->timeTag;
    
    /*! - Copy over the current state estimate into the 0th Sigma point and propagate by dt*/
	vCopy(configData->state, configData->numStates,
		&(configData->SP[0 * configData->numStates + 0]));
	sunlineStateProp(&(configData->SP[0 * configData->numStates + 0]),
        configData->dt);
    /*! - Scale that Sigma point by the appopriate scaling factor (Wm[0])*/
	vScale(configData->wM[0], &(configData->SP[0 * configData->numStates + 0]),
        configData->numStates, configData->xBar);
    /*! - Get the transpose of the sBar matrix because it is easier to extract Rows vs columns*/
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               sBarT);
    /*! - For each Sigma point, apply sBar-based error, propagate forward, and scale by Wm just like 0th.
          Note that we perform +/- sigma points simultaneously in loop to save loop values.*/
	for (i = 0; i<configData->countHalfSPs; i++)
	{
		Index = i + 1;
		spPtr = &(configData->SP[Index*configData->numStates]);
		vCopy(&sBarT[i*configData->numStates], configData->numStates, spPtr);
		vScale(configData->gamma, spPtr, configData->numStates, spPtr);
		vAdd(spPtr, configData->numStates, configData->state, spPtr);
		sunlineStateProp(spPtr, configData->dt);
		vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
		vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
		
		Index = i + 1 + configData->countHalfSPs;
        spPtr = &(configData->SP[Index*configData->numStates]);
        vCopy(&sBarT[i*configData->numStates], configData->numStates, spPtr);
        vScale(-configData->gamma, spPtr, configData->numStates, spPtr);
        vAdd(spPtr, configData->numStates, configData->state, spPtr);
        sunlineStateProp(spPtr, configData->dt);
        vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
        vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
	}
    /*! - Zero the AT matrix prior to assembly*/
    mSetZero(AT, (2 * configData->countHalfSPs + configData->numStates),
        configData->countHalfSPs);
	/*! - Assemble the AT matrix.  Note that this matrix is the internals of 
          the qr decomposition call in the source design documentation.  It is 
          the inside of equation 20 in that document*/
	for (i = 0; i<2 * configData->countHalfSPs; i++)
	{
		
        vScale(-1.0, configData->xBar, configData->numStates, aRow);
        vAdd(aRow, configData->numStates,
             &(configData->SP[(i+1)*configData->numStates]), aRow);
        vScale(sqrt(configData->wC[i+1]), aRow, configData->numStates, aRow);
		memcpy((void *)&AT[i*configData->numStates], (void *)aRow,
			configData->numStates*sizeof(double));
	}
    /*! - Pop the sQNoise matrix on to the end of AT prior to getting QR decomposition*/
	memcpy(&AT[2 * configData->countHalfSPs*configData->numStates],
		configData->sQnoise, configData->numStates*configData->numStates
        *sizeof(double));
    /*! - QR decomposition (only R computed!) of the AT matrix provides the new sBar matrix*/
    ukfQRDJustR(AT, 2 * configData->countHalfSPs + configData->numStates,
                configData->countHalfSPs, rAT);
    mCopy(rAT, configData->numStates, configData->numStates, sBarT);
    mTranspose(sBarT, configData->numStates, configData->numStates,
        configData->sBar);
    
    /*! - Shift the sBar matrix over by the xBar vector using the appropriate weight 
          like in equation 21 in design document.*/
    vScale(-1.0, configData->xBar, configData->numStates, xErr);
    vAdd(xErr, configData->numStates, &configData->SP[0], xErr);
    ukfCholDownDate(configData->sBar, xErr, configData->wC[0],
        configData->numStates, sBarUp);
    
    /*! - Save current sBar matrix, covariance, and state estimate off for further use*/
    mCopy(sBarUp, configData->numStates, configData->numStates, configData->sBar);
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
        configData->covar);
	mMultM(configData->sBar, configData->numStates, configData->numStates,
        configData->covar, configData->numStates, configData->numStates,
           configData->covar);
    vCopy(&(configData->SP[0]), configData->numStates, configData->state );
	
	configData->timeTag = updateTime;
}

/*! This method computes what the expected measurement vector is for each CSS 
    that is present on the spacecraft.  All data is transacted from the main 
    data structure for the model because there are many variables that would 
    have to be updated otherwise.
 @return void
 @param configData The configuration data associated with the CSS estimator

 */
void sunlineUKFMeasModel(SunlineUKFConfig *configData)
{
    
    double sensorNormal[3];

    int obsCounter = 0;
    /*! - Loop over all available coarse sun sensors and only use ones that meet validity threshold*/
    for(uint32_t i=0; i<configData->numCSSTotal; i++)
    {
        if(configData->cssSensorInBuffer.CosValue[i] > configData->sensorUseThresh)
        {
            /*! - For each valid measurement, copy observation value and compute expected obs value 
                  on a per sigma-point basis.*/
            v3Scale(configData->CBias[i], &(configData->cssNHat_B[i*3]), sensorNormal);
            configData->obs[obsCounter] = configData->cssSensorInBuffer.CosValue[i];
            for(int j=0; j<configData->countHalfSPs*2+1; j++)
            {
                configData->yMeas[obsCounter*(configData->countHalfSPs*2+1) + j] =
                    v3Dot(&(configData->SP[j*SKF_N_STATES]), sensorNormal);
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
void sunlineUKFMeasUpdate(SunlineUKFConfig *configData, double updateTime)
{
    double yBar[MAX_N_CSS_MEAS], syInv[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double kMat[SKF_N_STATES*MAX_N_CSS_MEAS];
    double xHat[SKF_N_STATES], sBarT[SKF_N_STATES*SKF_N_STATES], tempYVec[MAX_N_CSS_MEAS];
    double AT[(2 * SKF_N_STATES + MAX_N_CSS_MEAS)*MAX_N_CSS_MEAS], qChol[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double rAT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS], syT[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double sy[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];
    double updMat[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS], pXY[SKF_N_STATES*MAX_N_CSS_MEAS];
        
    /*! - Compute the valid observations and the measurement model for all observations*/
    sunlineUKFMeasModel(configData);
    
    /*! - Compute the value for the yBar parameter (note that this is equation 23 in the 
          time update section of the reference document*/
    vSetZero(yBar, configData->numObs);
    for(int i=0; i<configData->countHalfSPs*2+1; i++)
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
    for(int i=0; i<configData->countHalfSPs*2; i++)
    {
        vScale(-1.0, yBar, configData->numObs, tempYVec);
        vAdd(tempYVec, configData->numObs,
             &(configData->yMeas[(i+1)*configData->numObs]), tempYVec);
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
    ukfCholDecomp(configData->qObs, configData->numObs, configData->numObs, qChol);
    memcpy(&(AT[2*configData->countHalfSPs*configData->numObs]),
           qChol, configData->numObs*configData->numObs*sizeof(double));
    /*! - Perform QR decomposition (only R again) of the above matrix to obtain the 
          current Sy matrix*/
    ukfQRDJustR(AT, 2*configData->countHalfSPs+configData->numObs,
                configData->numObs, rAT);
    mCopy(rAT, configData->numObs, configData->numObs, syT);
    mTranspose(syT, configData->numObs, configData->numObs, sy);
    /*! - Shift the matrix over by the difference between the 0th SP-based measurement 
          model and the yBar matrix (cholesky down-date again)*/
    vScale(-1.0, yBar, configData->numObs, tempYVec);
    vAdd(tempYVec, configData->numObs, &(configData->yMeas[0]), tempYVec);
    ukfCholDownDate(sy, tempYVec, configData->wC[0],
                    configData->numObs, updMat);
    /*! - Shifted matrix represents the Sy matrix */
    mCopy(updMat, configData->numObs, configData->numObs, sy);
    mTranspose(sy, configData->numObs, configData->numObs, syT);

    /*! - Construct the Pxy matrix (equation 26) which multiplies the Sigma-point cloud 
          by the measurement model cloud (weighted) to get the total Pxy matrix*/
    mSetZero(pXY, configData->numStates, configData->numObs);
    for(int i=0; i<2*configData->countHalfSPs+1; i++)
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
    ukfUInv(syT, configData->numObs, configData->numObs, syInv);
    
    mMultM(pXY, configData->numStates, configData->numObs, syInv,
           configData->numObs, configData->numObs, kMat);
    ukfLInv(sy, configData->numObs, configData->numObs, syInv);
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
           configData->numObs, configData->numObs, pXY);
    mTranspose(pXY, configData->numStates, configData->numObs, pXY);
    /*! - For each column in the update matrix, perform a cholesky down-date on it to 
          get the total shifted S matrix (called sBar in internal parameters*/
    for(int i=0; i<configData->numObs; i++)
    {
        vCopy(&(pXY[i*configData->numStates]), configData->numStates, tempYVec);
        ukfCholDownDate(configData->sBar, tempYVec, -1.0, configData->numStates, sBarT);
        mCopy(sBarT, configData->numStates, configData->numStates,
            configData->sBar);
    }
    /*! - Compute equivalent covariance based on updated sBar matrix*/
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               configData->covar);
    mMultM(configData->sBar, configData->numStates, configData->numStates,
           configData->covar, configData->numStates, configData->numStates,
           configData->covar);
}
