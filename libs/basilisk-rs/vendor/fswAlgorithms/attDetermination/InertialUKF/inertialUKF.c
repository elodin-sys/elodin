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

#include <string.h>
#include <math.h>
#include "fswAlgorithms/attDetermination/InertialUKF/inertialUKF.h"
#include "architecture/utilities/ukfUtilities.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"

/*! This method creates the two moduel output messages.
 @return void
 @param configData The configuration data associated with the CSS WLS estimator
 @param moduleId The module identifier
 */
void SelfInit_inertialUKF(InertialUKFConfig *configData, int64_t moduleId)
{
    NavAttMsg_C_init(&configData->navStateOutMsg);
    InertialFilterMsg_C_init(&configData->filtDataOutMsg);
}


/*! This method resets the inertial inertial filter to an initial state and
 initializes the internal estimation matrices.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleId The module identifier
 */
void Reset_inertialUKF(InertialUKFConfig *configData, uint64_t callTime,
                      int64_t moduleId)
{
    
    size_t i;
    int32_t badUpdate=0; /* Negative badUpdate is faulty, */
    double tempMatrix[AKF_N_STATES*AKF_N_STATES];

    // check if the required input messages are included
    if (!RWArrayConfigMsg_C_isLinked(&configData->rwParamsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: inertialUKF.rwParamsInMsg wasn't connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->massPropsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: inertialUKF.massPropsInMsg wasn't connected.");
    }
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedsInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: inertialUKF.rwSpeedsInMsg wasn't connected.");
    }
    if (!AccDataMsg_C_isLinked(&configData->gyrBuffInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: inertialUKF.gyrBuffInMsg wasn't connected.");
    }

    /*! - Read static RW config data message and store it in module variables */
    configData->rwConfigParams = RWArrayConfigMsg_C_read(&configData->rwParamsInMsg);
    configData->localConfigData = VehicleConfigMsg_C_read(&configData->massPropsInMsg);

    /*! - Initialize filter parameters to max values */
    configData->timeTag = callTime*NANO2SEC;
    configData->dt = 0.0;
    configData->numStates = AKF_N_STATES;
    configData->countHalfSPs = AKF_N_STATES;
    configData->numObs = 3;
    configData->firstPassComplete = 0;
    configData->speedDt = 0.0;
    configData->timeWheelPrev = 0;
    
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
    for (i = 1; i<configData->countHalfSPs * 2 + 1; i++)
    {
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates +
                                             configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    vCopy(configData->stateInit, configData->numStates, configData->state);
    
    /*! - User a cholesky decomposition to obtain the sBar and sQnoise matrices for use in filter at runtime*/
    mCopy(configData->covarInit, configData->numStates, configData->numStates,
          configData->sBar);
    mCopy(configData->covarInit, configData->numStates, configData->numStates,
          configData->covar);
    mSetZero(tempMatrix, configData->numStates, configData->numStates);
    badUpdate += ukfCholDecomp(configData->sBar, (int32_t) configData->numStates,
                  (int32_t) configData->numStates, tempMatrix);
    
    badUpdate += ukfCholDecomp(configData->qNoise, (int32_t) configData->numStates,
                               (int32_t) configData->numStates, configData->sQnoise);

    mCopy(tempMatrix, configData->numStates, configData->numStates,
          configData->sBar);
    mTranspose(configData->sQnoise, configData->numStates,
               configData->numStates, configData->sQnoise);
    
    v3Copy(configData->state, configData->sigma_BNOut);
    v3Copy(&(configData->state[3]), configData->omega_BN_BOut);
    configData->timeTagOut = configData->timeTag;
    Read_STMessages(configData);

    if (badUpdate <0){
        _bskLog(configData->bskLogger, BSK_WARNING, "Reset method contained bad update");
    }
    return;
}

/*! This method reads in the messages from all available star trackers and orders them with respect to time of measurement
 @return void
 @param configData The configuration data associated with the CSS estimator
 */
void Read_STMessages(InertialUKFConfig *configData)
{
    uint64_t timeOfMsgWritten; /* [ns] Read time when the message was written*/
    int isWritten;      /* has the message been written */
    int bufferSTIndice; /* Local ST message to copy and organize  */
    int i;
    int j;
    
    for (i = 0; i < configData->STDatasStruct.numST; i++)
    {
        /*! - Read the input parsed CSS sensor data message*/
        configData->stSensorIn[i] = STAttMsg_C_read(&configData->STDatasStruct.STMessages[i].stInMsg);
        timeOfMsgWritten = STAttMsg_C_timeWritten(&configData->STDatasStruct.STMessages[i].stInMsg);

        /*! - Only mark valid size if message isn't stale*/
        isWritten = STAttMsg_C_isWritten(&configData->STDatasStruct.STMessages[i].stInMsg);
        configData->isFreshST[i] = timeOfMsgWritten != configData->ClockTimeST[i] ? isWritten : 0;
        configData->ClockTimeST[i] = timeOfMsgWritten;

        /*! - If the time tag from the measured data is new compared to previous step,
         propagate and update the filter*/
        configData->stSensorOrder[i] = i;
        /*! - Ensure that the time-tags we've received are put in time order*/
        for (j=i; j>0; j--)
        {
            if (configData->stSensorIn[configData->stSensorOrder[j]].timeTag <
                configData->stSensorIn[configData->stSensorOrder[j-1]].timeTag )
            {
                bufferSTIndice = configData->stSensorOrder[j];
                configData->stSensorOrder[j] =  configData->stSensorOrder[j-1];
                configData->stSensorOrder[j-1] = bufferSTIndice;
            }
        }
    }
    

}
/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_inertialUKF(InertialUKFConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    double newTimeTag;  /* [s] Local Time-tag variable*/
    uint64_t timeOfRWSpeeds; /* [ns] Read time for the RWs*/
    int32_t trackerValid; /* [-] Indicates whether the star tracker was valid*/
    double sigma_BNSum[3]; /* [-] Local MRP for propagated state*/
    InertialFilterMsgPayload inertialDataOutBuffer; /* [-] Output filter info*/
    AccDataMsgPayload gyrBuffer; /* [-] Buffer of IMU messages for gyro prop*/
    NavAttMsgPayload outputInertial;
    int i;
    
    // Reset update check to zero
    if (v3Norm(configData->state) > configData->switchMag) //Little extra margin
    {
        MRPswitch(configData->state, configData->switchMag, configData->state);
    }

    /* zero output buffer */
    outputInertial = NavAttMsg_C_zeroMsgPayload();

    /* read input messages */
    configData->localConfigData = VehicleConfigMsg_C_read(&configData->massPropsInMsg);
    gyrBuffer = AccDataMsg_C_read(&configData->gyrBuffInMsg);
    configData->rwSpeeds = RWSpeedMsg_C_read(&configData->rwSpeedsInMsg);
    timeOfRWSpeeds = RWSpeedMsg_C_timeWritten(&configData->rwSpeedsInMsg);
    Read_STMessages(configData);

    m33Inverse(RECAST3X3 configData->localConfigData.ISCPntB_B, configData->IInv);
    /*! - Handle initializing time in filter and discard initial messages*/
    if(configData->firstPassComplete == 0)
    {
        /*! - Set wheel speeds so that acceleration can be safely computed*/
        configData->rwSpeedPrev = configData->rwSpeeds;
        configData->timeWheelPrev = timeOfRWSpeeds;

        /*! - Loop through ordered time-tags and select largest valid one*/
        newTimeTag = 0.0;
        for (i=0; i<configData->STDatasStruct.numST; i++)
        {
            if(configData->isFreshST[configData->stSensorOrder[i]] &&
               configData->stSensorIn[configData->stSensorOrder[i]].timeTag*NANO2SEC > newTimeTag)
            {
                newTimeTag = configData->stSensorIn[configData->stSensorOrder[i]].timeTag*NANO2SEC;
                /*! - If any ST message is valid mark initialization complete*/
                configData->firstPassComplete = 1;
            }
        }
        /*! - If no ST messages were valid, return from filter and try again next frame*/
        if(configData->firstPassComplete == 0)
        {
            return;
        }
        configData->timeTag = newTimeTag;
    }

    configData->speedDt = (timeOfRWSpeeds - configData->timeWheelPrev)*NANO2SEC;
    configData->timeWheelPrev = timeOfRWSpeeds;
    
    inertialDataOutBuffer.numObs = 0;
    trackerValid = 0;
    for (i = 0; i < configData->STDatasStruct.numST; i++)
    {
        newTimeTag = configData->stSensorIn[configData->stSensorOrder[i]].timeTag * NANO2SEC;
        int isFresh = configData->isFreshST[configData->stSensorOrder[i]];

        /*! - If the star tracker has provided a new message compared to last time,
              update the filter to the new measurement*/
        if(newTimeTag >= configData->timeTag && isFresh)
        {
            trackerValid = 1;
            if((newTimeTag - configData->timeTag) > configData->maxTimeJump
                && configData->maxTimeJump > 0)
            {
                configData->timeTag = newTimeTag - configData->maxTimeJump;
                _bskLog(configData->bskLogger, BSK_WARNING, "Large jump in state time that was set to max.");
            }
            trackerValid += inertialUKFTimeUpdate(configData, newTimeTag);
            trackerValid += inertialUKFMeasUpdate(configData, configData->stSensorOrder[i]);
            inertialDataOutBuffer.numObs += trackerValid;
        }
    }
    /*! - If current clock time is further ahead than the measured time, then
     propagate to this current time-step*/
    newTimeTag = callTime*NANO2SEC;
    if(trackerValid < 1)
    {
        /*! - If no star tracker measurement was available, propagate the state
         on the gyro measurements received since the last ST update.  Note
         that the rate estimate is just smoothed gyro data in this case*/
        
        /*! - Assemble the aggregrate rotation from the gyro buffer*/
        inertialUKFAggGyrData(configData, configData->timeTagOut,
                              newTimeTag, &gyrBuffer);
        /*! - Propagate the attitude quaternion with the aggregate rotation*/
        addMRP(configData->sigma_BNOut, configData->aggSigma_b2b1,
               sigma_BNSum);
        /*! - Switch the MRPs if necessary*/
        if (v3Norm(sigma_BNSum) > configData->switchMag) //Little extra margin
        {
            MRPswitch(sigma_BNSum, configData->switchMag, sigma_BNSum);
        }
        v3Copy(sigma_BNSum, configData->sigma_BNOut);
        /*! - Rate estimate in this case is simply the low-pass filtered
         gyro data.  This is likely much noisier than the time-update
         solution*/
        for(i=0; i<3; i++)
        {
            configData->omega_BN_BOut[i] = configData->gyroFilt[i].currentState;
        }
        configData->timeTagOut = configData->gyrAggTimeTag;
        
    }
    else
    {
        /*! - If we are already at callTime just copy the states over without
         change*/
        v3Copy(configData->state, configData->sigma_BNOut);
        v3Copy(&(configData->state[3]), configData->omega_BN_BOut);
        configData->timeTagOut = configData->timeTag;
    }
    
    /*! - Write the inertial estimate into the copy of the navigation message structure*/
    v3Copy(configData->sigma_BNOut, outputInertial.sigma_BN);
    v3Copy(configData->omega_BN_BOut, outputInertial.omega_BN_B);
    outputInertial.timeTag = configData->timeTagOut;
	
    NavAttMsg_C_write(&outputInertial, &configData->navStateOutMsg, moduleID, callTime);
    
    /*! - Populate the filter states output buffer and write the output message*/
    inertialDataOutBuffer.timeTag = configData->timeTag;
    memmove(inertialDataOutBuffer.covar, configData->covar,
            AKF_N_STATES*AKF_N_STATES*sizeof(double));
    memmove(inertialDataOutBuffer.state, configData->state, AKF_N_STATES*sizeof(double));
    InertialFilterMsg_C_write(&inertialDataOutBuffer, &configData->filtDataOutMsg, moduleID, callTime);
    configData->rwSpeedPrev = configData->rwSpeeds;
    return;
}

/*! This method propagates a inertial state vector forward in time.  Note 
    that the calling parameter is updated in place to save on data copies.
	@return void
    @param configData The configuration data associated with this module
    @param stateInOut The state that is propagated
    @param dt Time step (s)
*/
void inertialStateProp(InertialUKFConfig *configData, double *stateInOut, double dt)
{

    double sigmaDot[3];
    double BMatrix[3][3];
    double torqueTotal[3];
    double wheelAccel;
    double torqueSingle[3];
    double angAccelTotal[3];
    int i;
    
    /*! - Convert the state derivative (body rate) to sigmaDot and propagate
          the attitude MRPs*/
    BmatMRP(stateInOut, BMatrix);
    m33Scale(0.25, BMatrix, BMatrix);
    m33MultV3(BMatrix, &(stateInOut[3]), sigmaDot);
    v3Scale(dt, sigmaDot, sigmaDot);
    v3Add(stateInOut, sigmaDot, stateInOut);
    
    /*! - Assemble the total torque from the reaction wheels to get the forcing
     function from any wheels present*/
    v3SetZero(torqueTotal);
    for(i=0; i<configData->rwConfigParams.numRW; i++)
    {
        if(configData->speedDt == 0.0)
        {
            continue;
        }
        wheelAccel = configData->rwSpeeds.wheelSpeeds[i]-
            configData->rwSpeedPrev.wheelSpeeds[i];
        wheelAccel /= configData->speedDt/configData->rwConfigParams.JsList[i];
        v3Scale(wheelAccel, &(configData->rwConfigParams.GsMatrix_B[i*3]), torqueSingle);
        v3Subtract(torqueTotal, torqueSingle, torqueTotal);
    }
    /*! - Get the angular acceleration and propagate the state forward (euler prop)*/
    m33MultV3(configData->IInv, torqueTotal, angAccelTotal);
    v3Scale(dt, angAccelTotal, angAccelTotal);
    v3Add(&(stateInOut[3]), angAccelTotal, &(stateInOut[3]));
	return;
}

/*! This method performs the time update for the inertial kalman filter.
     It propagates the sigma points forward in time and then gets the current 
	 covariance and state estimates.
	 @return void
     @param configData The configuration data associated with the CSS estimator
     @param updateTime The time that we need to fix the filter to (seconds)
*/
int inertialUKFTimeUpdate(InertialUKFConfig *configData, double updateTime)
{
    size_t i;
    int Index, k;
	double sBarT[AKF_N_STATES*AKF_N_STATES];
	double xComp[AKF_N_STATES], AT[(2 * AKF_N_STATES + AKF_N_STATES)*AKF_N_STATES];
	double aRow[AKF_N_STATES], rAT[AKF_N_STATES*AKF_N_STATES], xErr[AKF_N_STATES]; 
	double sBarUp[AKF_N_STATES*AKF_N_STATES];
	double *spPtr;
    double procNoise[AKF_N_STATES*AKF_N_STATES];
    int32_t badUpdate=0;
    
	configData->dt = updateTime - configData->timeTag;
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    
    mSetZero(rAT, AKF_N_STATES, AKF_N_STATES);
    mCopy(configData->sQnoise, AKF_N_STATES, AKF_N_STATES, procNoise);
    /*! - Copy over the current state estimate into the 0th Sigma point and propagate by dt*/
	vCopy(configData->state, configData->numStates,
		&(configData->SP[0 * configData->numStates + 0]));
	inertialStateProp(configData, &(configData->SP[0]),
        configData->dt);
    /*! - Scale that Sigma point by the appopriate scaling factor (Wm[0])*/
	vScale(configData->wM[0], &(configData->SP[0]),
        configData->numStates, configData->xBar);
    /*! - Get the transpose of the sBar matrix because it is easier to extract Rows vs columns*/
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               sBarT);
    /*! - For each Sigma point, apply sBar-based error, propagate forward, and scale by Wm just like 0th.
          Note that we perform +/- sigma points simultaneously in loop to save loop values.*/
	for (i = 0; i<configData->countHalfSPs; i++)
	{
        /*! - Adding covariance columns from sigma points*/
		Index = (int) i + 1;
		spPtr = &(configData->SP[Index* (int) configData->numStates]);
		vCopy(&sBarT[i* (int) configData->numStates], configData->numStates, spPtr);
		vScale(configData->gamma, spPtr, configData->numStates, spPtr);
		vAdd(spPtr, configData->numStates, configData->state, spPtr);
		inertialStateProp(configData, spPtr, configData->dt);
		vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
		vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
		/*! - Subtracting covariance columns from sigma points*/
		Index = (int) i + 1 + (int) configData->countHalfSPs;
        spPtr = &(configData->SP[Index* (int) configData->numStates]);
        vCopy(&sBarT[i* (int) configData->numStates], configData->numStates, spPtr);
        vScale(-configData->gamma, spPtr, configData->numStates, spPtr);
        vAdd(spPtr, configData->numStates, configData->state, spPtr);
        inertialStateProp(configData, spPtr, configData->dt);
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
             &(configData->SP[(i+1)* (int) configData->numStates]), aRow);
        /*Check sign of wC to know if the sqrt will fail*/
        if (configData->wC[i+1]<=0){
            inertialUKFCleanUpdate(configData);
            return -1;}
        vScale(sqrt(configData->wC[i+1]), aRow, configData->numStates, aRow);
		memcpy((void *)&AT[i* (int) configData->numStates], (void *)aRow,
			configData->numStates*sizeof(double));
        
	}
   /*! - Scale sQNoise matrix depending on the dt*/
    for (k=0;k<3;k++){
        procNoise[k*AKF_N_STATES+k] *= configData->dt*configData->dt/2;
        procNoise[(k+3)*AKF_N_STATES+(k+3)] *= configData->dt;
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
    vCopy(&(configData->SP[0]), configData->numStates, configData->state);
	
    if (badUpdate<0){
        inertialUKFCleanUpdate(configData);
        return(-1);}
    else{
        configData->timeTag = updateTime;
    }
    return(0);
}

/*! This method computes what the expected measurement vector is for each CSS 
    that is present on the spacecraft.  All data is transacted from the main 
    data structure for the model because there are many variables that would 
    have to be updated otherwise.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param currentST current star tracker state

 */
void inertialUKFMeasModel(InertialUKFConfig *configData, int currentST)
{
    double quatTranspose[4];
    double quatMeas[4];
    double EPSum[4];
    double mrpSum[3];
    size_t i;
    
    /*! This math seems more difficult than it should be, but there is a method.
        The input MRP may or may not be in the same "shadow" set as the state estimate.
        So, if they are different in terms of light/shadow, you have to get them 
        to the same representation otherwise your residuals will show 360 degree 
        errors.  Which is not ideal.  So that's why it is so blessed complicated.  
        The measurement is shadowed into the same representation as the state.*/
    MRP2EP(configData->state, quatTranspose);
    v3Scale(-1.0, &(quatTranspose[1]), &(quatTranspose[1]));
    MRP2EP(configData->stSensorIn[currentST].MRP_BdyInrtl, quatMeas);
    addEP(quatTranspose, quatMeas, EPSum);
    EP2MRP(EPSum, mrpSum);
    if (v3Norm(mrpSum) > 1.0)
    {
        MRPshadow(configData->stSensorIn[currentST].MRP_BdyInrtl,
                  configData->stSensorIn[currentST].MRP_BdyInrtl);
    }
    
    /*! - The measurement model is the same as the states since the star tracker 
          measures the inertial attitude directly.*/
    for(i=0; i<configData->countHalfSPs*2+1; i++)
    {
        v3Copy(&(configData->SP[i*AKF_N_STATES]), &(configData->yMeas[i*3]));
    }
    
    v3Copy(configData->stSensorIn[currentST].MRP_BdyInrtl, configData->obs);
    configData->numObs = 3;
    
}

/*! This method aggregates the input gyro data into a combined total quaternion 
    rotation to push the state forward by.  This information is stored in the 
    main data structure for use in the propagation routines.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param prevTime [s] Previous time step
 @param propTime The time that we need to fix the filter to (seconds)
 @param gyrData The gyro measurements that we are going to accumulate forward into time
 */
void inertialUKFAggGyrData(InertialUKFConfig *configData, double prevTime,
    double propTime, AccDataMsgPayload *gyrData)
{
    uint32_t minFutInd;  /* [-] Index in buffer that is the oldest new meas*/
    int i, j;
    double minFutTime;   /* [s] smallest future measurement time-tag*/
    double measTime;     /* [s] measurement time*/
    /*! Note that the math here is tortured to avoid the issues of adding
          PRVs together.  That is numerically problematic, so we convert to 
          euler parameters (quaternions) and add those*/
    double ep_BpropB0[4], ep_B1B0[4], epTemp[4], omeg_BN_B[3], prvTemp[3];
    double dt;
    
    minFutInd = 0;
    minFutTime = -1;
    /*! - Loop through the entire gyro buffer to find the first index that is 
          in the future compared to prevTime*/
    for(i=0; i<MAX_ACC_BUF_PKT; i++)
    {
        measTime = gyrData->accPkts[i].measTime*NANO2SEC;
        if(measTime > prevTime && (measTime < minFutTime || minFutTime < 0.0))
        {
            minFutInd = (uint32_t) i;
            minFutTime = measTime;
        }
    }
    /*! - Initialize the propagated euler parameters and time*/
    v4SetZero(ep_BpropB0);
    ep_BpropB0[0] = 1.0;
    i=0;
    measTime = prevTime;
    /*! - Loop through buffer for all valid measurements to assemble the 
          composite rotation since the previous time*/
    while(minFutTime > prevTime && i<MAX_ACC_BUF_PKT)
    {
        dt = minFutTime - measTime;
        /*! - Treat rates scaled by dt as a PRV (small angle approximation)*/
        v3Copy(gyrData->accPkts[minFutInd].gyro_B, omeg_BN_B);
        v3Scale(dt, omeg_BN_B, prvTemp);
        
        /*! - Convert the PRV to euler parameters and add that delta-rotation 
              to the running sum (ep_BpropB0)*/
        PRV2EP(prvTemp, ep_B1B0);
        v4Copy(ep_BpropB0, epTemp);
        addEP(epTemp, ep_B1B0, ep_BpropB0);
        configData->gyrAggTimeTag = minFutTime;
        i++;
        /*! - Prepare for the next measurement and set time-tags for termination*/
        measTime = minFutTime;
        /*% operator used because gyro buffer is a ring-buffer and this operator 
            wraps the index back to zero when we overflow.*/
        minFutInd = (minFutInd + 1)%MAX_ACC_BUF_PKT;
        minFutTime = gyrData->accPkts[minFutInd].measTime*NANO2SEC;
        /*! - Apply low-pass filter to gyro measurements to get smoothed body rate*/
        for(j=0; j<3; j++)
        {
            lowPassFilterSignal(omeg_BN_B[j], &(configData->gyroFilt[j]));
        }
    }
    /*! - Saved the measurement count and convert the euler parameters to MRP
          as that is our filter representation*/
    configData->numUsedGyros = (uint32_t) i;
    EP2MRP(ep_BpropB0, configData->aggSigma_b2b1);
    
    return;
}

/*! This method performs the measurement update for the inertial kalman filter.
 It applies the observations in the obs vectors to the current state estimate and
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param currentST Current star tracker state
 */
int inertialUKFMeasUpdate(InertialUKFConfig *configData, int currentST)
{
    uint32_t i;
    double yBar[3], syInv[3*3];
    double kMat[AKF_N_STATES*3];
    double xHat[AKF_N_STATES], Ucol[AKF_N_STATES], sBarT[AKF_N_STATES*AKF_N_STATES], tempYVec[3];
    double AT[(2 * AKF_N_STATES + 3)*3], qChol[3*3];
    double rAT[3*3], syT[3*3];
    double sy[3*3];
    double updMat[3*3], pXY[AKF_N_STATES*3], Umat[AKF_N_STATES*3];
    int32_t badUpdate=0;
    
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    /*! - Compute the valid observations and the measurement model for all observations*/
    inertialUKFMeasModel(configData, currentST);

    mSetZero(rAT, 3, 3);
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
        if (configData->wC[i+1]<0){return -1;}
        vScale(sqrt(configData->wC[i+1]), tempYVec, configData->numObs, tempYVec);
        memcpy(&(AT[i*configData->numObs]), tempYVec,
               configData->numObs*sizeof(double));
    }
    
    /*! - This is the square-root of the Rk matrix which we treat as the Cholesky
        decomposition of the observation variance matrix constructed for our number 
        of observations*/
    badUpdate += ukfCholDecomp(configData->STDatasStruct.STMessages[currentST].noise, (int32_t) configData->numObs, (int32_t) configData->numObs, qChol);
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
        inertialUKFCleanUpdate(configData);
        return(-1);}
    return(0);
}

/*! This method cleans the filter states after a bad upadate on the fly.
 It removes the potentially corrupted previous estimates and puts the filter
 back to a working state.
 @return void
 @param configData The configuration data associated with the CSS estimator
 */
void inertialUKFCleanUpdate(InertialUKFConfig *configData){
    size_t i;
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
    for (i = 1; i<configData->countHalfSPs * 2 + 1; i++)
    {
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates +
                                             configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    return;
}

