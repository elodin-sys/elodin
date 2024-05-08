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

#include "fswAlgorithms/attDetermination/CSSEst/cssWlsEst.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for theCSS WLS estimator.
 @return void
 @param configData The configuration data associated with the CSS WLS estimator
 @param moduleID The module identifier
 */
void SelfInit_cssWlsEst(CSSWLSConfig *configData, int64_t moduleID)
{
    NavAttMsg_C_init(&configData->navStateOutMsg);
    if (SunlineFilterMsg_C_isLinked(&configData->cssWLSFiltResOutMsg)) {
        SunlineFilterMsg_C_init(&configData->cssWLSFiltResOutMsg);
    }
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the guidance module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Reset_cssWlsEst(CSSWLSConfig *configData, uint64_t callTime, int64_t moduleID)
{

    // check that required messages have been included
    if (!CSSConfigMsg_C_isLinked(&configData->cssConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: cssWIsEst.cssConfigInMsg wasn't connected.");
    }
    if (!CSSArraySensorMsg_C_isLinked(&configData->cssDataInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: cssWIsEst.cssDataInMsg wasn't connected.");
    }

    configData->cssConfigInBuffer = CSSConfigMsg_C_read(&configData->cssConfigInMsg);
    if (configData->cssConfigInBuffer.nCSS > MAX_N_CSS_MEAS) {
        _bskLog(configData->bskLogger, BSK_ERROR, "cssWIsEst.cssDataInMsg.nCSS must not be greater than "
                                                  "MAX_N_CSS_MEAS value.");
    }

    configData->priorSignalAvailable = 0;
    v3SetZero(configData->dOld);
    
    configData->filtStatus.numObs = 0;
    configData->filtStatus.timeTag = 0.0;
    v3SetZero(configData->filtStatus.state);
    vSetZero(configData->filtStatus.postFitRes, MAX_N_CSS_MEAS);
    

    /* Reset the prior time flag state.
     If zero, control time step not evaluated on the first function call */
    configData->priorTime = 0;

    return;
}

/*! This method computes the post-fit residuals for the WLS estimate.  Note that 
    everything has to have been allocated appropriately as this function operates 
    directly on the arrays.
    @return void
    @param cssMeas The measured values for the CSS sensors
    @param cssConfig The CSS configuration information
    @param wlsEst The WLS estimate computed for the CSS measurements
    @param cssResiduals The measurement residuals output by this function
*/
void computeWlsResiduals(double *cssMeas, CSSConfigMsgPayload *cssConfig,
                         double *wlsEst, double *cssResiduals)
{
    double cssDotProd;
    
    memset(cssResiduals, 0x0, cssConfig->nCSS*sizeof(double));
    /*! The method loops through the sensors and performs: */
    for(uint32_t i=0; i<cssConfig->nCSS; i++)
    {
        /*! -# A dot product between the computed estimate with each sensor normal */
        cssDotProd = v3Dot(wlsEst, cssConfig->cssVals[i].nHat_B);
        cssDotProd = cssDotProd > 0.0 ? cssDotProd : 0.0; /*CSS values can't be negative!*/
        /*! -# A subtraction between that post-fit measurement estimate and the actual measurement*/
        cssResiduals[i] = cssMeas[i] - cssDotProd;
        /*! -# This populates the post-fit residuals*/
    }
    
}

/*! This method computes a least squares fit with the given parameters.  It
 treats the inputs as though they were double dimensioned arrays but they
 are all singly dimensioned for ease of use
 @return success indicator (0 for good, 1 for fail)
 @param numActiveCss The count on input measurements
 @param H The predicted pointing vector for each measurement
 @param W the weighting matrix for the set of measurements
 @param y the observation vector for the valid sensors
 @param x The output least squares fit for the observations
 */
int computeWlsmn(int numActiveCss, double *H, double *W,
                 double *y, double x[3])
{
    double m22[2*2];
    double m32[3*2];
    int status = 0;
    double  m33[3*3];
    double  m33_2[3*3];
    double  m3N[3*MAX_NUM_CSS_SENSORS];
    double  m3N_2[3*MAX_NUM_CSS_SENSORS];
    uint32_t i;
    
    /*! - If we only have one sensor, output best guess (cone of possiblities)*/
    if(numActiveCss == 1) {
        /* Here's a guess.  Do with it what you will. */
        for(i = 0; i < 3; i=i+1) {
            x[i] = H[0*MAX_NUM_CSS_SENSORS+i] * y[0];
        }
    } else if(numActiveCss == 2) { /*! - If we have two, then do a 2x2 fit */
        
        /*!   -# Find minimum norm solution */
        mMultMt(H, 2, 3, H, 2, 3, m22);
        status = m22Inverse(RECAST2x2 m22, RECAST2x2 m22);
        mtMultM(H, 2, 3, m22, 2, 2, m32);
        /*!   -# Multiply the Ht(HHt)^-1 by the observation vector to get fit*/
        mMultV(m32, 3, 2, y, x);
    } else if(numActiveCss > 2) {/*! - If we have more than 2, do true LSQ fit*/
        /*!    -# Use the weights to compute (HtWH)^-1HW*/
        mtMultM(H, (size_t) numActiveCss, 3, W, (size_t) numActiveCss, (size_t) numActiveCss, m3N);
        mMultM(m3N, 3, (size_t) numActiveCss, H, (size_t) numActiveCss, 3, m33);
        status = m33Inverse(RECAST3X3 m33, RECAST3X3 m33_2);
        mMultMt(m33_2, 3, 3, H, (size_t) numActiveCss, 3, m3N);
        mMultM(m3N, 3, (size_t) numActiveCss, W, (size_t) numActiveCss, (size_t) numActiveCss, m3N_2);
        /*!    -# Multiply the LSQ matrix by the obs vector for best fit*/
        mMultV(m3N_2, 3, (size_t) numActiveCss, y, x);
    }
    
    return(status);
}

/*! This method takes the parsed CSS sensor data and outputs an estimate of the
 sun vector in the ADCS body frame
 @return void
 @param configData The configuration data associated with the CSS estimator
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identifier
 */
void Update_cssWlsEst(CSSWLSConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    CSSArraySensorMsgPayload InputBuffer;        /* CSS measurements */
    double H[MAX_NUM_CSS_SENSORS*3];             /* The predicted pointing vector for each measurement */
    double y[MAX_NUM_CSS_SENSORS];               /* Measurements */
    double W[MAX_NUM_CSS_SENSORS*MAX_NUM_CSS_SENSORS];  /* Matrix of measurement weights */
    int status = 0;                              /* Quality of the module estimate */
    double dOldDotNew;                           /* Intermediate value for dot product between new and old estimates for rate estimation */
    double dHatNew[3];                           /* New normalized sun heading estimate */
    double dHatOld[3];                           /* Prior normalized sun heading estimate */
    double  dt;                                  /* [s] Control update period */
    NavAttMsgPayload sunlineOutBuffer;               /* Output Nav message*/
    
    /* Zero output message*/
    sunlineOutBuffer = NavAttMsg_C_zeroMsgPayload();

    /*! Message Read and Setup*/
    /*! - Read the input parsed CSS sensor data message*/
    InputBuffer = CSSArraySensorMsg_C_read(&configData->cssDataInMsg);

    /*! - Compute control update time */
    if (configData->priorTime == 0) {
        dt = 0.0;
    } else {
        dt = (callTime - configData->priorTime) * NANO2SEC;
    }
    configData->priorTime = callTime;
    
    /* - Zero the observed active CSS count*/
    configData->numActiveCss = 0;
    
    /*! - Loop over the maximum number of sensors to check for good measurements */
    /*! -# Isolate if measurement is good */
    /*! -# Set body vector for this measurement */
    /*! -# Get measurement value into observation vector */
    /*! -# Set inverse noise matrix */
    /*! -# increase the number of valid observations */
    /*! -# Otherwise just continue */
    for(uint32_t i=0; i<configData->cssConfigInBuffer.nCSS; i = i+1)
    {
        if(InputBuffer.CosValue[i] > configData->sensorUseThresh)
        {
            v3Scale(configData->cssConfigInBuffer.cssVals[i].CBias,
                configData->cssConfigInBuffer.cssVals[i].nHat_B, &H[configData->numActiveCss*3]);
            y[configData->numActiveCss] = InputBuffer.CosValue[i];
            configData->numActiveCss = configData->numActiveCss + 1;
            
        }
    }
    
    /*! Estimation Steps*/
    configData->filtStatus = SunlineFilterMsg_C_zeroMsgPayload();

    if(configData->numActiveCss == 0) /*! - If there is no sun, just quit*/
    {
        /*! + If no CSS got a strong enough signal.  Sun estimation is not possible.  Return the zero vector instead */
        v3SetZero(sunlineOutBuffer.vehSunPntBdy);       /* zero the sun heading to indicate now CSS info is available */
        v3SetZero(sunlineOutBuffer.omega_BN_B);         /* zero the rate measure */
        configData->priorSignalAvailable = 0;                       /* reset the prior heading estimate flag */
        computeWlsResiduals(InputBuffer.CosValue, &configData->cssConfigInBuffer,
                            sunlineOutBuffer.vehSunPntBdy, configData->filtStatus.postFitRes);
    } else {
        /*! - If at least one CSS got a strong enough signal.  Proceed with the sun heading estimation */
        /*! -# Configuration option to weight the measurements, otherwise set
         weighting matrix to identity*/
        if(configData->useWeights > 0)
        {
            mDiag(y, configData->numActiveCss, W);
        }
        else
        {
            mSetIdentity(W, configData->numActiveCss, configData->numActiveCss);
        }
        /*! -# Get least squares fit for sun pointing vector*/
        status = computeWlsmn((int) configData->numActiveCss, H, W, y,
                              sunlineOutBuffer.vehSunPntBdy);
        computeWlsResiduals(InputBuffer.CosValue, &configData->cssConfigInBuffer,
                            sunlineOutBuffer.vehSunPntBdy, configData->filtStatus.postFitRes);

        v3Normalize(sunlineOutBuffer.vehSunPntBdy, sunlineOutBuffer.vehSunPntBdy);

        /*! -# Estimate the inertial angular velocity from the rate of the sun heading measurements */
        if (configData->priorSignalAvailable && dt > 0.0) {
            v3Normalize(sunlineOutBuffer.vehSunPntBdy, dHatNew);
            v3Normalize(configData->dOld, dHatOld);
            v3Cross(dHatNew, dHatOld, sunlineOutBuffer.omega_BN_B);
            v3Normalize(sunlineOutBuffer.omega_BN_B,sunlineOutBuffer.omega_BN_B);
            /* compute principal rotation angle between sun heading measurements */
            dOldDotNew = v3Dot(dHatNew,dHatOld);
            if (dOldDotNew > 1.0) dOldDotNew = 1.0;
            if (dOldDotNew < -1.0) dOldDotNew = -1.0;
            v3Scale(safeAcos(dOldDotNew)/dt, sunlineOutBuffer.omega_BN_B, sunlineOutBuffer.omega_BN_B);
        } else {
            configData->priorSignalAvailable = 1;
        }
        /*! -# Store the sun heading estimate */
        v3Copy(sunlineOutBuffer.vehSunPntBdy, configData->dOld);
    }

    /*! Residual Computation */
    /*! - If the residual fit output message is set, then compute the residuals and stor them in the output message */
    if (SunlineFilterMsg_C_isLinked(&configData->cssWLSFiltResOutMsg)) {
        configData->filtStatus.numObs = (int) configData->numActiveCss;
        configData->filtStatus.timeTag = (double) (callTime*NANO2SEC);
        v3Copy(sunlineOutBuffer.vehSunPntBdy, configData->filtStatus.state);
        SunlineFilterMsg_C_write(&configData->filtStatus, &configData->cssWLSFiltResOutMsg, moduleID, callTime);

    }
    /*! Writing Outputs */
    if(status > 0) /*! - If the status from the WLS computation is erroneous, populate the output messages with zeros*/
    {
        /* An error was detected while attempting to compute the sunline direction */
        v3SetZero(sunlineOutBuffer.vehSunPntBdy);       /* zero the sun heading to indicate anomaly  */
        v3SetZero(sunlineOutBuffer.omega_BN_B);         /* zero the rate measure */
        configData->priorSignalAvailable = 0;                       /* reset the prior heading estimate flag */
    }
    /*! - If the status from the WLS computation good, populate the output messages with the computed data*/
    NavAttMsg_C_write(&sunlineOutBuffer, &configData->navStateOutMsg, moduleID, callTime);
    return;
}
