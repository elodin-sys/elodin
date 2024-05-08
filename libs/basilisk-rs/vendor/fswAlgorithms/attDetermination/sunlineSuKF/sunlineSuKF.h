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

#ifndef _SUNLINE_UKF_H_
#define _SUNLINE_UKF_H_

#include <stdint.h>
#include <string.h>

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
#include "cMsgCInterface/SunlineFilterMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"

#include "architecture/utilities/bskLogging.h"



/*! structure containing the fitting parameters
 */
typedef struct {
    double cssRelScale;                          //!< Relative scale factor for this CSS
    double cssKellPow;                           //!< CSS kelly estimate power
    double cssKellFact;                          //!< CSS kelly scale factor
}SunlineSuKFCFit;

/*!@brief Data structure for CSS Switch unscented kalman filter estimator.
 */
typedef struct {
    NavAttMsg_C navStateOutMsg;                     /*!< The name of the output message*/
    SunlineFilterMsg_C filtDataOutMsg;              /*!< The name of the output filter data message*/
    CSSArraySensorMsg_C cssDataInMsg;               /*!< The name of the Input message*/
    CSSConfigMsg_C cssConfigInMsg;                  /*!< [-] The name of the CSS configuration message*/

	size_t numStates;                           //!< [-] Number of states for this filter
	size_t countHalfSPs;                        //!< [-] Number of sigma points over 2
	size_t numObs;                              //!< [-] Number of measurements this cycle
	double beta;                                //!< [-] Beta parameter for filter
	double alpha;                               //!< [-] Alpha parameter for filter
	double kappa;                               //!< [-] Kappa parameter for filter
	double lambdaVal;                           //!< [-] Lambda parameter for filter
	double gamma;                               //!< [-] Gamma parameter for filter
    double qObsVal;                             //!< [-] CSS instrument noise parameter

	double dt;                                  //!< [s] seconds since last data epoch
	double timeTag;                             //!< [s]  Time tag for statecovar/etc

    double bVec_B[SKF_N_STATES_HALF];           //!< [-] current vector of the b frame used to make frame
    double switchTresh;                         //!< [-]  Threshold for switching frames
    
    double stateInit[SKF_N_STATES_SWITCH];      //!< [-] State to initialize filter to
    double state[SKF_N_STATES_SWITCH];          //!< [-] State estimate for time TimeTag
    double statePrev[SKF_N_STATES_SWITCH];      //!< [-] Previous state logged for clean
    
	double wM[2 * SKF_N_STATES_SWITCH + 1];     //!< [-] Weighting vector for sigma points
	double wC[2 * SKF_N_STATES_SWITCH + 1];     //!< [-] Weighting vector for sigma points

    double sBar[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];         //!< [-] Time updated covariance
    double sBarPrev[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];     //!< [-] Time updated covariance logged for clean
    double covarInit[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];    //!< [-] covariance to init to
	double covar[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];        //!< [-] covariance
    double covarPrev[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];    //!< [-] Covariance logged for clean
    double xBar[SKF_N_STATES_SWITCH];                             //!< [-] Current mean state estimate

	double obs[MAX_N_CSS_MEAS];                                   //!< [-] Observation vector for frame
	double yMeas[MAX_N_CSS_MEAS*(2*SKF_N_STATES_SWITCH+1)];       //!< [-] Measurement model data
    double postFits[MAX_N_CSS_MEAS];                              //!< [-] PostFit residuals
    
	double SP[(2*SKF_N_STATES_SWITCH+1)*SKF_N_STATES_SWITCH];     //!< [-]    sigma point matrix

	double qNoise[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];       //!< [-] process noise matrix
	double sQnoise[SKF_N_STATES_SWITCH*SKF_N_STATES_SWITCH];      //!< [-] cholesky of Qnoise

	double qObs[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS]; //!< [-] Maximally sized obs noise matrix
    
    double cssNHat_B[MAX_NUM_CSS_SENSORS*3];    //!< [-] CSS normal vectors converted over to body
    double CBias[MAX_NUM_CSS_SENSORS];          //!< [-] CSS individual calibration coefficients
    SunlineSuKFCFit kellFits[MAX_NUM_CSS_SENSORS]; //!< [-] Curve fit components for CSS sensors

    uint32_t numActiveCss;                      //!< -- Number of currently active CSS sensors
    uint32_t numCSSTotal;                       //!< [-] Count on the number of CSS we have on the spacecraft
    double sensorUseThresh;                     //!< -- Threshold below which we discount sensors
	NavAttMsgPayload outputSunline;                 //!< -- Output sunline estimate data
    CSSArraySensorMsgPayload cssSensorInBuffer;     //!< [-] CSS sensor data read in from message bus
    uint32_t filterInitialized;                 //!< [-] Flag indicating if filter has been init or not

    BSKLogger *bskLogger;                         //!< BSK Logging
}SunlineSuKFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_sunlineSuKF(SunlineSuKFConfig *configData, int64_t moduleID);
    void Update_sunlineSuKF(SunlineSuKFConfig *configData, uint64_t callTime,
        int64_t moduleID);
	void Reset_sunlineSuKF(SunlineSuKFConfig *configData, uint64_t callTime,
		int64_t moduleID);
	int sunlineSuKFTimeUpdate(SunlineSuKFConfig *configData, double updateTime);
    int sunlineSuKFMeasUpdate(SunlineSuKFConfig *configData, double updateTime);
	void sunlineStateProp(double *stateInOut,  double *b_vec, double dt);
    void sunlineSuKFMeasModel(SunlineSuKFConfig *configData);
    void sunlineSuKFComputeDCM_BS(double sunheading[SKF_N_STATES_HALF], double bVec[SKF_N_STATES_HALF], double *dcm);
    void sunlineSuKFSwitch(double *bVec_B, double *states, double *covar);
    void sunlineSuKFCleanUpdate(SunlineSuKFConfig *configData);

#ifdef __cplusplus
}
#endif


#endif
