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
#include <stdlib.h>
#include <math.h>
#include "pixelLineBiasUKF.h"
#include "architecture/utilities/ukfUtilities.h"

/*! This method creates the two moduel output messages.
 @return void
 @param configData The configuration data associated with the OD filter
 @param moduleId The ID associated with the configData
 */
void SelfInit_pixelLineBiasUKF(PixelLineBiasUKFConfig *configData, int64_t moduleId)
{
    NavTransMsg_C_init(&configData->navStateOutMsg);
    PixelLineFilterMsg_C_init(&configData->filtDataOutMsg);
}


/*! This method resets the relative OD filter to an initial state and
 initializes the internal estimation matrices.
 @return void
 @param configData The configuration data associated with the OD filter
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleId The ID associated with the configData
 */
void Reset_pixelLineBiasUKF(PixelLineBiasUKFConfig *configData, uint64_t callTime,
                       int64_t moduleId)
{
    // check if the required message has not been connected
    if (!OpNavCirclesMsg_C_isLinked(&configData->circlesInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineBiasUKF.circlesInMsg wasn't connected.");
    }
    if (!CameraConfigMsg_C_isLinked(&configData->cameraConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineBiasUKF.cameraConfigInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->attInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineBiasUKF.attInMsg wasn't connected.");
    }

    size_t i;
    int32_t badUpdate=0; /* Negative badUpdate is faulty, */
    double tempMatrix[PIXLINE_N_STATES*PIXLINE_N_STATES];
    
    /*! - Initialize filter parameters to max values */
    configData->timeTag = callTime*NANO2SEC;
    configData->dt = 0.0;
    configData->numStates = PIXLINE_N_STATES;
    configData->countHalfSPs = PIXLINE_N_STATES;
    configData->numObs = PIXLINE_N_MEAS;
    configData->firstPassComplete = 0;
    configData->planetId = configData->planetIdInit;
    
    /*! - Ensure that all internal filter matrices are zeroed*/
    vSetZero(configData->obs, configData->numObs);
    vSetZero(configData->wM, configData->countHalfSPs * 2 + 1);
    vSetZero(configData->wC, configData->countHalfSPs * 2 + 1);
    mSetZero(configData->sBar, configData->numStates, configData->numStates);
    mSetZero(configData->SP, configData->countHalfSPs * 2 + 1,
             configData->numStates);
    mSetZero(configData->sQnoise, configData->numStates, configData->numStates);
    mSetZero(configData->measNoise, PIXLINE_N_MEAS, PIXLINE_N_MEAS);
    
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
        configData->wM[i] = 1.0 / 2.0*1.0 / (configData->numStates + configData->lambdaVal);
        configData->wC[i] = configData->wM[i];
    }
    
    vCopy(configData->stateInit, configData->numStates, configData->state);
    v6Scale(1E-3, configData->state, configData->state); // Convert to km
    /*! - User a cholesky decomposition to obtain the sBar and sQnoise matrices for use in filter at runtime*/
    mCopy(configData->covarInit, configData->numStates, configData->numStates,
          configData->sBar);
    vScale(1E-6, configData->sBar, PIXLINE_DYN_STATES*PIXLINE_N_STATES + PIXLINE_DYN_STATES, configData->sBar); // Convert to km
    mCopy(configData->covarInit, configData->numStates, configData->numStates,
          configData->covar);
    vScale(1E-6, configData->covar, PIXLINE_DYN_STATES*PIXLINE_N_STATES + PIXLINE_DYN_STATES, configData->covar); // Convert to km
    
    mSetZero(tempMatrix, configData->numStates, configData->numStates);
    badUpdate += ukfCholDecomp(configData->sBar, (int) configData->numStates,
                               (int) configData->numStates, tempMatrix);
    
    badUpdate += ukfCholDecomp(configData->qNoise, (int) configData->numStates,
                               (int) configData->numStates, configData->sQnoise);
    
    mCopy(tempMatrix, configData->numStates, configData->numStates,
          configData->sBar);
    mTranspose(configData->sQnoise, configData->numStates,
               configData->numStates, configData->sQnoise);
    
    configData->timeTagOut = configData->timeTag;
    
    if (badUpdate <0){
        _bskLog(configData->bskLogger, BSK_WARNING, "Reset method contained bad update");
    }
    return;
}

/*! This method takes the relative position measurements and outputs an estimate of the
 spacecraft states in the intertial frame.
 @return void
 @param configData The configuration data associated with the OD filter
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleId The ID associated with the configData
 */
void Update_pixelLineBiasUKF(PixelLineBiasUKFConfig *configData, uint64_t callTime,
                        int64_t moduleId)
{
    double newTimeTag = 0.0;  /* [s] Local Time-tag variable*/
    int32_t trackerValid; /* [-] Indicates whether the star tracker was valid*/
    double yBar[PIXLINE_N_MEAS], tempYVec[PIXLINE_N_MEAS];
    uint64_t i;
    int computePostFits;
    PixelLineFilterMsgPayload opNavOutBuffer; /* [-] Output filter info*/
    NavTransMsgPayload outputRelOD;
    OpNavCirclesMsgPayload inputCircles;
    configData->moduleId = (int) moduleId;
    
    computePostFits = 0;
    v3SetZero(configData->postFits);

    opNavOutBuffer = PixelLineFilterMsg_C_zeroMsgPayload();
    outputRelOD = NavTransMsg_C_zeroMsgPayload();


    /*! - read input messages */
    inputCircles = OpNavCirclesMsg_C_read(&configData->circlesInMsg);
    configData->cameraSpecs = CameraConfigMsg_C_read(&configData->cameraConfigInMsg);
    configData->attInfo = NavAttMsg_C_read(&configData->attInMsg);

    /*! - Handle initializing time in filter and discard initial messages*/
    trackerValid = 0;
    /*! - If the time tag from the measured data is new compared to previous step,
     propagate and update the filter*/
    newTimeTag = NavAttMsg_C_timeWritten(&configData->attInMsg) * NANO2SEC;
    if(newTimeTag >= configData->timeTag && NavAttMsg_C_isWritten(&configData->attInMsg) && inputCircles.valid ==1)
    {
        configData->circlesInBuffer = inputCircles;
        configData->planetId = (int) inputCircles.planetIds[0];
        pixelLineBiasUKFTimeUpdate(configData, newTimeTag);
        pixelLineBiasUKFMeasUpdate(configData);
        computePostFits = 1;
    }
    /*! - If current clock time is further ahead than the measured time, then
     propagate to this current time-step*/
    newTimeTag = callTime*NANO2SEC;
    if(newTimeTag > configData->timeTag)
    {
        pixelLineBiasUKFTimeUpdate(configData, newTimeTag);
    }

    
    /*! - The post fits are y - ybar if a measurement was read, if observations are zero, do not compute post fit residuals*/
    if(computePostFits == 1){
        /*! - Compute Post Fit Residuals, first get Y (eq 22) using the states post fit*/
        pixelLineBiasUKFMeasModel(configData);
        
        /*! - Compute the value for the yBar parameter (equation 23)*/
        vSetZero(yBar, configData->numObs);
        for(i=0; i<configData->countHalfSPs*2+1; i++)
        {
            vCopy(&(configData->yMeas[i*configData->numObs]), configData->numObs,
                  tempYVec);
            vScale(configData->wM[i], tempYVec, configData->numObs, tempYVec);
            vAdd(yBar, configData->numObs, tempYVec, yBar);
        }
        mSubtract(configData->obs, PIXLINE_N_MEAS, 1, yBar, configData->postFits);
    }
    
   
    /*! - Write the relative OD estimate into the copy of the navigation message structure*/
    v3Copy(configData->state, outputRelOD.r_BN_N);
    outputRelOD.timeTag = configData->timeTag;
    v3Scale(1E3, outputRelOD.r_BN_N, outputRelOD.r_BN_N); // Convert to m
    v3Copy(&configData->state[3], outputRelOD.v_BN_N);
    v3Scale(1E3, outputRelOD.v_BN_N, outputRelOD.v_BN_N); // Convert to m
    outputRelOD.timeTag = configData->timeTagOut;
    NavTransMsg_C_write(&outputRelOD, &configData->navStateOutMsg, moduleId, callTime);
    
    /*! - Populate the filter states output buffer and write the output message*/
    opNavOutBuffer.timeTag = configData->timeTag;
    memmove(opNavOutBuffer.covar, configData->covar,
            PIXLINE_N_STATES*PIXLINE_N_STATES*sizeof(double));
    memmove(opNavOutBuffer.state, configData->state, PIXLINE_N_STATES*sizeof(double));
    memmove(opNavOutBuffer.postFitRes, configData->postFits, PIXLINE_N_MEAS*sizeof(double));
    v6Scale(1E3, opNavOutBuffer.state, opNavOutBuffer.state); // Convert to m
    vScale(1E6, opNavOutBuffer.covar, PIXLINE_DYN_STATES*PIXLINE_N_STATES+PIXLINE_DYN_STATES, opNavOutBuffer.covar); // Convert to m
    PixelLineFilterMsg_C_write(&opNavOutBuffer, &configData->filtDataOutMsg, moduleId, callTime);
    
    return;
}

/*! This method propagates a relative OD state vector forward in time.  Note
 that the calling parameter is updated in place to save on data copies.
 @return void
 @param configData The configuration data associated with the OD filter
 @param stateInOut The state that is propagated
 @param dt Time step (s)
 */
void relODStateProp(PixelLineBiasUKFConfig *configData, double *stateInOut, double dt)
{
    
    double muPlanet;
    double k1[PIXLINE_DYN_STATES], k2[PIXLINE_DYN_STATES], k3[PIXLINE_DYN_STATES], k4[PIXLINE_DYN_STATES];
    double states1[PIXLINE_DYN_STATES], states2[PIXLINE_DYN_STATES], states3[PIXLINE_DYN_STATES];
    if(configData->planetId ==1){muPlanet = MU_EARTH;} //in km
    if(configData->planetId ==2){muPlanet = MU_MARS;} //in km
    if(configData->planetId ==3){muPlanet = MU_JUPITER;} //in km
    
    /*! Start RK4 */
    /*! - Compute k1 */
    pixelLineBiasUKFTwoBodyDyn(stateInOut, muPlanet, &k1[0]);
    vScale(dt/2, k1, PIXLINE_DYN_STATES, k1); // k1 is now k1/2
    /*! - Compute k2 */
    vAdd(stateInOut, PIXLINE_DYN_STATES, k1, states1);
    pixelLineBiasUKFTwoBodyDyn(states1, muPlanet, &k2[0]);
    vScale(dt/2, k2, PIXLINE_DYN_STATES, k2); // k2 is now k2/2
    /*! - Compute k3 */
    vAdd(stateInOut, PIXLINE_DYN_STATES, k2, states2);
    pixelLineBiasUKFTwoBodyDyn(states2, muPlanet, &k3[0]);
    vScale(dt, k3, PIXLINE_DYN_STATES, k3);
    /*! - Compute k4 */
    vAdd(stateInOut, PIXLINE_DYN_STATES, k3, states3);
    pixelLineBiasUKFTwoBodyDyn(states3, muPlanet, &k4[0]);
    vScale(dt, k4, PIXLINE_DYN_STATES, k4);
    /*! - Gather all terms with proper scales */
    vScale(1./3., k1, PIXLINE_DYN_STATES, k1); // k1 is now k1/6
    vScale(2./3., k2, PIXLINE_DYN_STATES, k2); // k2 is now k2/3
    vScale(1./3., k3, PIXLINE_DYN_STATES, k3); // k3 is now k2/3
    vScale(1./6., k4, PIXLINE_DYN_STATES, k4); // k4 is now k2/6
    
    vAdd(stateInOut, PIXLINE_DYN_STATES, k1, stateInOut);
    vAdd(stateInOut, PIXLINE_DYN_STATES, k2, stateInOut);
    vAdd(stateInOut, PIXLINE_DYN_STATES, k3, stateInOut);
    vAdd(stateInOut, PIXLINE_DYN_STATES, k4, stateInOut);
    
    return;
}

/*! Function for two body dynamics solvers in order to use in the RK4. Only two body dynamics is used currently, but SRP, Solar Gravity, spherical harmonics can be added here.
 @return double Next state
 @param state The starting state
 @param muPlanet planet gravity constant
 @param stateDeriv derivative of state set
 */
void pixelLineBiasUKFTwoBodyDyn(double state[PIXLINE_DYN_STATES], double muPlanet, double *stateDeriv)
{
    double rNorm;
    double dvdt[3];
    
    rNorm = v3Norm(state);
    v3Copy(&state[3], stateDeriv);
    v3Copy(state, dvdt);
    v3Scale(-muPlanet/pow(rNorm, 3), dvdt, &stateDeriv[3]);
    return;
}

/*! This method performs the time update for the relative OD kalman filter.
 It propagates the sigma points forward in time and then gets the current
 covariance and state estimates.
 @return void
 @param configData The configuration data associated with the OD filter
 @param updateTime The time that we need to fix the filter to (seconds)
 */
int pixelLineBiasUKFTimeUpdate(PixelLineBiasUKFConfig *configData, double updateTime)
{
    uint64_t i, Index;
    double sBarT[PIXLINE_N_STATES*PIXLINE_N_STATES]; // Sbar transpose (chol decomp of covar)
    double xComp[PIXLINE_N_STATES], AT[(2 * PIXLINE_N_STATES + PIXLINE_N_STATES)*PIXLINE_N_STATES]; // Intermediate state, process noise chol decomp
    double aRow[PIXLINE_N_STATES], rAT[PIXLINE_N_STATES*PIXLINE_N_STATES], xErr[PIXLINE_N_STATES]; //Row of A mat, R of QR decomp of A, state error
    double sBarUp[PIXLINE_N_STATES*PIXLINE_N_STATES]; // S bar cholupdate
    double *spPtr; //sigma point intermediate varaible
    double procNoise[PIXLINE_N_STATES*PIXLINE_N_STATES]; //process noise
    int32_t badUpdate=0;
    
    configData->dt = updateTime - configData->timeTag;
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    
    /*! - Read the planet ID from the message*/
    if(configData->planetId == 0)
    {
      _bskLog(configData->bskLogger, BSK_ERROR, "Need a planet to navigate");
    }

    mCopy(configData->sQnoise, PIXLINE_N_STATES, PIXLINE_N_STATES, procNoise);
    /*! - Copy over the current state estimate into the 0th Sigma point and propagate by dt*/
    vCopy(configData->state, configData->numStates,
          &(configData->SP[0 * configData->numStates + 0]));
    relODStateProp(configData, &(configData->SP[0]),
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
        Index = i + 1;
        spPtr = &(configData->SP[Index*configData->numStates]);
        vCopy(&sBarT[i*configData->numStates], configData->numStates, spPtr);
        vScale(configData->gamma, spPtr, configData->numStates, spPtr);
        vAdd(spPtr, configData->numStates, configData->state, spPtr);
        relODStateProp(configData, spPtr, configData->dt);
        vScale(configData->wM[Index], spPtr, configData->numStates, xComp);
        vAdd(xComp, configData->numStates, configData->xBar, configData->xBar);
        /*! - Subtracting covariance columns from sigma points*/
        Index = i + 1 + configData->countHalfSPs;
        spPtr = &(configData->SP[Index*configData->numStates]);
        vCopy(&sBarT[i*configData->numStates], configData->numStates, spPtr);
        vScale(-configData->gamma, spPtr, configData->numStates, spPtr);
        vAdd(spPtr, configData->numStates, configData->state, spPtr);
        relODStateProp(configData, spPtr, configData->dt);
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
        /*Check sign of wC to know if the sqrt will fail*/
        if (configData->wC[i+1]<=0){
            pixelLineBiasUKFCleanUpdate(configData);
            return -1;}
        vScale(sqrt(configData->wC[i+1]), aRow, configData->numStates, aRow);
        memcpy((void *)&AT[i*configData->numStates], (void *)aRow,
               configData->numStates*sizeof(double));
        
    }
    /*! - Pop the sQNoise matrix on to the end of AT prior to getting QR decomposition*/
    memcpy(&AT[2 * configData->countHalfSPs*configData->numStates],
           procNoise, configData->numStates*configData->numStates
           *sizeof(double));
    /*! - QR decomposition (only R computed!) of the AT matrix provides the new sBar matrix*/
    ukfQRDJustR(AT, (int) (2 * configData->countHalfSPs + configData->numStates),
                (int) configData->countHalfSPs, rAT);
    
    mCopy(rAT, configData->numStates, configData->numStates, sBarT);
    mTranspose(sBarT, configData->numStates, configData->numStates,
               configData->sBar);
    
    /*! - Shift the sBar matrix over by the xBar vector using the appropriate weight
     like in equation 21 in design document.*/
    vScale(-1.0, configData->xBar, configData->numStates, xErr);
    vAdd(xErr, configData->numStates, &configData->SP[0], xErr);
    badUpdate += ukfCholDownDate(configData->sBar, xErr, configData->wC[0],
                                 (int) configData->numStates, sBarUp);
    
    
    /*! - Save current sBar matrix, covariance, and state estimate off for further use*/
    mCopy(sBarUp, configData->numStates, configData->numStates, configData->sBar);
    mTranspose(configData->sBar, configData->numStates, configData->numStates,
               configData->covar);
    mMultM(configData->sBar, configData->numStates, configData->numStates,
           configData->covar, configData->numStates, configData->numStates,
           configData->covar);
    vCopy(&(configData->SP[0]), configData->numStates, configData->state);
    
    if (badUpdate<0){
        pixelLineBiasUKFCleanUpdate(configData);
        return(-1);}
    else{
        configData->timeTag = updateTime;
    }
    return(0);
}

/*! This method computes the measurement model.  Given that the data is coming from
 the pixelLine Converter, the transformation has already taken place from pixel data to spacecraft position.
 @return void
 @param configData The configuration data associated with the OD filter
 */
void pixelLineBiasUKFMeasModel(PixelLineBiasUKFConfig *configData)
{
    size_t i, j;
    double dcm_CN[3][3], dcm_CB[3][3], dcm_BN[3][3];
    double reCentered[2], rNorm, denom, planetRad;
    double r_C[3];

    v3Set(configData->circlesInBuffer.circlesCenters[0], configData->circlesInBuffer.circlesCenters[1], configData->circlesInBuffer.circlesRadii[0], configData->obs);

    MRP2C(configData->cameraSpecs.sigma_CB, dcm_CB);
    MRP2C(configData->attInfo.sigma_BN, dcm_BN);
    m33MultM33(dcm_CB, dcm_BN, dcm_CN);
    double X, Y;
    double pX, pY;
    /* compute sensorSize/focalLength = 2*tan(FOV/2) */
    pX = 2.*tan(configData->cameraSpecs.fieldOfView*configData->cameraSpecs.resolution[0]/configData->cameraSpecs.resolution[1]/2.0);
    pY = 2.*tan(configData->cameraSpecs.fieldOfView/2.0);
    X = pX/configData->cameraSpecs.resolution[0];
    Y = pY/configData->cameraSpecs.resolution[1];
    
    if(configData->circlesInBuffer.planetIds[0] > 0){
        if(configData->circlesInBuffer.planetIds[0] ==1){
            planetRad = REQ_EARTH;//in km
        }
        if(configData->circlesInBuffer.planetIds[0] ==2){
            planetRad = REQ_MARS;//in km
        }
        if(configData->circlesInBuffer.planetIds[0] ==3){
            planetRad = REQ_JUPITER;//in km
        }
    }
    
    for(j=0; j<configData->countHalfSPs*2+1; j++)
    {
        double centers[2], r_N_bar[3], radius=0;
        v2SetZero(centers);
        v3SetZero(r_N_bar);
        
        v3Copy(&configData->SP[j*configData->numStates], r_N_bar);
        rNorm = v3Norm(r_N_bar);
        
        m33MultV3(dcm_CN, r_N_bar, r_C);
        v3Scale(-1./r_C[2], r_C, r_C);
        
        /*! - Find pixel size using camera specs */
        reCentered[0] = r_C[0]/X;
        reCentered[1] = r_C[1]/Y;
        
        centers[0] = reCentered[0] + configData->cameraSpecs.resolution[0]/2 - 0.5;
        centers[1] = reCentered[1] + configData->cameraSpecs.resolution[1]/2 - 0.5;
        
        denom = planetRad/rNorm;
        radius = tan(safeAsin(denom)) / X;
        if (j==0){
            v2Subtract(centers, configData->obs, &configData->obs[3]);
            configData->obs[5] = radius - configData->obs[2];
        }
        for(i=0; i<3; i++){
            configData->obs[i+3] = round(configData->obs[i+3]);
            if (i<2){
                configData->yMeas[i*(configData->countHalfSPs*2+1) + j] = centers[i] - configData->SP[j*configData->numStates+PIXLINE_DYN_STATES + i];
            }
            if (i==2){
                configData->yMeas[i*(configData->countHalfSPs*2+1) + j] = radius - configData->SP[j*configData->numStates+PIXLINE_DYN_STATES + i];
            }
            configData->yMeas[(i+PIXLINE_N_MEAS/2)*(configData->countHalfSPs*2+1) + j] = configData->SP[j*configData->numStates+ PIXLINE_DYN_STATES + i];
        }
    }
    /*! - yMeas matrix was set backwards deliberately so we need to transpose it through*/
    mTranspose(configData->yMeas, PIXLINE_N_MEAS, configData->countHalfSPs*2+1,
               configData->yMeas);
    
}

/*! This method performs the measurement update for the kalman filter.
 It applies the observations in the obs vectors to the current state estimate and
 updates the state/covariance with that information.
 @return void
 @param configData The configuration data associated with the OD filter
 */
int pixelLineBiasUKFMeasUpdate(PixelLineBiasUKFConfig *configData)
{
    uint32_t i;
    double yBar[PIXLINE_N_MEAS], syInv[PIXLINE_N_MEAS*PIXLINE_N_MEAS]; //measurement, Sy inv
    double kMat[PIXLINE_N_STATES*PIXLINE_N_MEAS], cholNoise[PIXLINE_N_MEAS*PIXLINE_N_MEAS];//Kalman Gain, chol decomp of noise
    double xHat[PIXLINE_N_STATES], Ucol[PIXLINE_N_STATES], sBarT[PIXLINE_N_STATES*PIXLINE_N_STATES], tempYVec[PIXLINE_N_MEAS];// state error, U column eq 28, intermediate variables
    double AT[(2 * PIXLINE_N_STATES + PIXLINE_N_MEAS)*PIXLINE_N_MEAS]; //Process noise matrix
    double rAT[PIXLINE_N_MEAS*PIXLINE_N_MEAS], syT[PIXLINE_N_MEAS*PIXLINE_N_MEAS]; //QR R decomp, Sy transpose
    double sy[PIXLINE_N_MEAS*PIXLINE_N_MEAS]; // Chol of covariance
    double updMat[PIXLINE_N_MEAS*PIXLINE_N_MEAS], pXY[PIXLINE_N_STATES*PIXLINE_N_MEAS], Umat[PIXLINE_N_STATES*PIXLINE_N_MEAS]; // Intermediate variable, covariance eq 26, U eq 28
    int32_t badUpdate=0;
    
    vCopy(configData->state, configData->numStates, configData->statePrev);
    mCopy(configData->sBar, configData->numStates, configData->numStates, configData->sBarPrev);
    mCopy(configData->covar, configData->numStates, configData->numStates, configData->covarPrev);
    /*! - Compute the valid observations and the measurement model for all observations*/
    pixelLineBiasUKFMeasModel(configData);
    
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
    mSetZero(configData->measNoise, PIXLINE_N_MEAS, PIXLINE_N_MEAS);
    mSetSubMatrix(configData->circlesInBuffer.uncertainty, 3, 3, configData->measNoise, PIXLINE_N_MEAS, PIXLINE_N_MEAS, 3,3);
    mSetSubMatrix(configData->circlesInBuffer.uncertainty, 3, 3, configData->measNoise, PIXLINE_N_MEAS, PIXLINE_N_MEAS, 0, 0);
    badUpdate += ukfCholDecomp(configData->measNoise, PIXLINE_N_MEAS, PIXLINE_N_MEAS, cholNoise);
    memcpy(&(AT[2*configData->countHalfSPs*configData->numObs]),
           cholNoise, configData->numObs*configData->numObs*sizeof(double));
    /*! - Perform QR decomposition (only R again) of the above matrix to obtain the
     current Sy matrix*/
    ukfQRDJustR(AT, (int) (2*configData->countHalfSPs+configData->numObs),
                (int) configData->numObs, rAT);
    
    mCopy(rAT, configData->numObs, configData->numObs, syT);
    mTranspose(syT, configData->numObs, configData->numObs, sy);
    /*! - Shift the matrix over by the difference between the 0th SP-based measurement
     model and the yBar matrix (cholesky down-date again)*/
    vScale(-1.0, yBar, configData->numObs, tempYVec);
    vAdd(tempYVec, configData->numObs, &(configData->yMeas[0]), tempYVec);
    badUpdate += ukfCholDownDate(sy, tempYVec, configData->wC[0],
                                 (int) configData->numObs, updMat);
    
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
    ukfUInv(syT, (int) configData->numObs, (int) configData->numObs, syInv);
    
    mMultM(pXY, configData->numStates, configData->numObs, syInv,
           configData->numObs, configData->numObs, kMat);
    ukfLInv(sy, (int) configData->numObs, (int) configData->numObs, syInv);
    mMultM(kMat, configData->numStates, configData->numObs, syInv,
           configData->numObs, configData->numObs, kMat);
    
    
    /*! - Difference the yBar and the observations to get the observed error and
     multiply by the Kalman Gain to get the state update.  Add the state update
     to the state to get the updated state value (equation 27).*/
    vSubtract(configData->obs, configData->numObs, yBar, tempYVec);
    mMultM(kMat, configData->numStates, configData->numObs, tempYVec,
           configData->numObs, 1, xHat);
    vAdd(configData->state, configData->numStates, xHat, configData->state);
    for (i=6;i<9;i++){
        configData->state[i] = round(configData->state[i]);
    }
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
        badUpdate += ukfCholDownDate(configData->sBar, Ucol, -1.0, (int) configData->numStates, sBarT);
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
        pixelLineBiasUKFCleanUpdate(configData);
        return(-1);}
    return(0);
}

/*! This method cleans the filter states after a bad upadate on the fly.
 It removes the potentially corrupted previous estimates and puts the filter
 back to a working state.
 @return void
 @param configData The configuration data associated with the OD filter
 */
void pixelLineBiasUKFCleanUpdate(PixelLineBiasUKFConfig *configData){
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

