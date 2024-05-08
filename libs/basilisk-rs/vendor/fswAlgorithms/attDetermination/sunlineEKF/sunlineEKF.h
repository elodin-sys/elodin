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

#ifndef _SUNLINE_EKF_H_
#define _SUNLINE_EKF_H_

#include <stdint.h>
#include <string.h>

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
#include "cMsgCInterface/SunlineFilterMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"

#include "architecture/utilities/bskLogging.h"





/*! @brief Top level structure for the CSS-based Extended Kalman Filter.
 Used to estimate the sun state in the vehicle body frame. */

typedef struct {
    NavAttMsg_C navStateOutMsg;                     /*!< The name of the output message*/
    SunlineFilterMsg_C filtDataOutMsg;              /*!< The name of the output filter data message*/
    CSSArraySensorMsg_C cssDataInMsg;               /*!< The name of the Input message*/
    CSSConfigMsg_C cssConfigInMsg;                  /*!< [-] The name of the CSS configuration message*/

    double qObsVal;               /*!< [-] CSS instrument noise parameter*/
    double qProcVal;               /*!< [-] Process noise parameter*/

	double dt;                     /*!< [s] seconds since last data epoch */
	double timeTag;                /*!< [s]  Time tag for state/covar */

	double state[SKF_N_STATES];        /*!< [-] State estimate for time TimeTag*/
    double x[SKF_N_STATES];             /*!< State errors */
    double xBar[SKF_N_STATES];            /*!< [-] Current mean time updated state estimate*/
	double covarBar[SKF_N_STATES*SKF_N_STATES];         /*!< [-] Time updated covariance */
	double covar[SKF_N_STATES*SKF_N_STATES];        /*!< [-] covariance */
    double stateTransition[SKF_N_STATES*SKF_N_STATES];        /*!< [-] State Transtion Matrix */
    double kalmanGain[SKF_N_STATES*MAX_N_CSS_MEAS];    /*!< Kalman Gain */

    double dynMat[SKF_N_STATES*SKF_N_STATES];        /*!< [-] Dynamics Matrix, A */
    double measMat[MAX_N_CSS_MEAS*SKF_N_STATES];        /*!< [-] Measurement Matrix H*/
    
	double obs[MAX_N_CSS_MEAS];          /*!< [-] Observation vector for frame*/
	double yMeas[MAX_N_CSS_MEAS];        /*!< [-] Linearized measurement model data */

	double procNoise[SKF_N_STATES/2*SKF_N_STATES/2];       /*!< [-] process noise matrix */
	double measNoise[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];  /*!< [-] Maximally sized obs noise matrix*/
    double postFits[MAX_N_CSS_MEAS];  /*!< [-] PostFit residuals */
    
    double cssNHat_B[MAX_NUM_CSS_SENSORS*3];     /*!< [-] CSS normal vectors converted over to body*/
    double CBias[MAX_NUM_CSS_SENSORS];       /*!< [-] CSS individual calibration coefficients */

    size_t numStates;                /*!< [-] Number of states for this filter*/
    int numObs;                   /*!< [-] Number of measurements this cycle */
    size_t numActiveCss;   /*!< -- Number of currently active CSS sensors*/
    size_t numCSSTotal;    /*!< [-] Count on the number of CSS we have on the spacecraft*/
    double sensorUseThresh;  /*!< -- Threshold below which we discount sensors*/
    double eKFSwitch;       /*!< -- Max covariance element after which the filter switches to an EKF update*/
	NavAttMsgPayload outputSunline;   /*!< -- Output sunline estimate data */
    CSSArraySensorMsgPayload cssSensorInBuffer; /*!< [-] CSS sensor data read in from message bus*/

    BSKLogger *bskLogger;   //!< BSK Logging
}sunlineEKFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_sunlineEKF(sunlineEKFConfig *configData, int64_t moduleID);
	void Reset_sunlineEKF(sunlineEKFConfig *configData, uint64_t callTime,
		int64_t moduleID);
    void Update_sunlineEKF(sunlineEKFConfig *configData, uint64_t callTime,
                           int64_t moduleID);
	void sunlineTimeUpdate(sunlineEKFConfig *configData, double updateTime);
    void sunlineMeasUpdate(sunlineEKFConfig *configData, double updateTime);
	void sunlineStateSTMProp(double dynMat[SKF_N_STATES*SKF_N_STATES], double dt, double *stateInOut, double *stateTransition);
    
    void sunlineHMatrixYMeas(double states[SKF_N_STATES], int numCSS, double cssSensorCos[MAX_N_CSS_MEAS], double sensorUseThresh, double cssNHat_B[MAX_NUM_CSS_SENSORS*3], double CBias[MAX_NUM_CSS_SENSORS], double *obs, double *yMeas, int *numObs, double *measMat);
    
    void sunlineKalmanGain(double covarBar[SKF_N_STATES*SKF_N_STATES], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double qObsVal, int numObs, double *kalmanGain);
    
    void sunlineDynMatrix(double stateInOut[SKF_N_STATES], double dt, double *dynMat);
    
    void sunlineCKFUpdate(double xBar[SKF_N_STATES], double kalmanGain[SKF_N_STATES*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES*SKF_N_STATES], double qObsVal, int numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double *x, double *covar);
    
    void sunlineEKFUpdate(double kalmanGain[SKF_N_STATES*MAX_N_CSS_MEAS], double covarBar[SKF_N_STATES*SKF_N_STATES], double qObsVal, int numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*SKF_N_STATES], double *states, double *x, double *covar);
    
#ifdef __cplusplus
}
#endif


#endif
