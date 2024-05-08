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

#ifndef _SUNLINE_EKF_H_
#define _SUNLINE_EKF_H_

#include <stdint.h>

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
#include "cMsgCInterface/SunlineFilterMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <string.h>



/*! @brief Top level structure for the CSS-based Switch Extended Kalman Filter.
 Used to estimate the sun state in the vehicle body frame. */
typedef struct {
    NavAttMsg_C navStateOutMsg;                     /*!< The name of the output message*/
    SunlineFilterMsg_C filtDataOutMsg;              /*!< The name of the output filter data message*/
    CSSArraySensorMsg_C cssDataInMsg;               /*!< The name of the Input message*/
    CSSConfigMsg_C cssConfigInMsg;                  /*!< [-] The name of the CSS configuration message*/

    double qObsVal;               /*!< [-] CSS instrument noise parameter*/
    double qProcVal;               /*!< [-] Process noise parameter*/

	double dt;                     /*!< [s] seconds since last data epoch */
	double timeTag;                /*!< [s]  Time tag for statecovar/etc */

    double bVec_B[SKF_N_STATES_HALF];       /*!< [-] current vector of the b frame used to make Switch frame */
    double switchTresh;             /*!< [-]  Cosine of angle between singularity and S-frame. If close to 1, the threshold for switching frames is lower. If closer to 0.5 singularity is more largely avoided but switching is more frequent  */

	double state[EKF_N_STATES_SWITCH];        /*!< [-] State estimate for time TimeTag*/
    double x[EKF_N_STATES_SWITCH];             /*!< State errors */
    double xBar[EKF_N_STATES_SWITCH];            /*!< [-] Current mean state estimate*/
	double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];         /*!< [-] Time updated covariance */
	double covar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];        /*!< [-] covariance */
    double stateTransition[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];        /*!< [-] State transition Matrix */
    double kalmanGain[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS];    /*!< Kalman Gain */

    double dynMat[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];        /*!< [-] Dynamics Matrix, A */
    double measMat[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH];        /*!< [-] Measurement Matrix, H*/
    double W_BS[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH];        /*!< [-] Switch Matrix to bring states and covariance to new S-frame when switch occurs*/
    
	double obs[MAX_N_CSS_MEAS];          /*!< [-] Observation vector for frame*/
	double yMeas[MAX_N_CSS_MEAS];        /*!< [-] Linearized measurement model data */
    double postFits[MAX_N_CSS_MEAS];  /*!< [-] PostFit residuals */

	double procNoise[(EKF_N_STATES_SWITCH-3)*(EKF_N_STATES_SWITCH-3)];       /*!< [-] process noise matrix */
	double measNoise[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];  /*!< [-] Maximally sized obs noise matrix*/
    
    double cssNHat_B[MAX_NUM_CSS_SENSORS*3];     /*!< [-] CSS normal vectors converted over to body*/
    uint32_t numStates;                /*!< [-] Number of states for this filter*/
    size_t numObs;                   /*!< [-] Number of measurements this cycle */
    uint32_t numActiveCss;   /*!< -- Number of currently active CSS sensors*/
    uint32_t numCSSTotal;    /*!< [-] Count on the number of CSS we have on the spacecraft*/
    double sensorUseThresh;  /*!< -- Threshold below which we discount sensors*/
    double eKFSwitch;       /*!< -- Max covariance element after which the filter switches to an EKF*/
	NavAttMsgPayload outputSunline;   /*!< -- Output sunline estimate data */
    CSSArraySensorMsgPayload cssSensorInBuffer; /*!< [-] CSS sensor data read in from message bus*/

    BSKLogger *bskLogger;   //!< BSK Logging
}sunlineSEKFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_sunlineSEKF(sunlineSEKFConfig *configData, int64_t moduleID);
	void Reset_sunlineSEKF(sunlineSEKFConfig *configData, uint64_t callTime,
		int64_t moduleID);
    void Update_sunlineSEKF(sunlineSEKFConfig *configData, uint64_t callTime,
                           int64_t moduleID);
	void sunlineTimeUpdate(sunlineSEKFConfig *configData, double updateTime);
    void sunlineMeasUpdate(sunlineSEKFConfig *configData, double updateTime);
	void sunlineStateSTMProp(double dynMat[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double bVec[SKF_N_STATES], double dt, double *stateInOut, double *stateTransition);
    
    void sunlineHMatrixYMeas(double states[EKF_N_STATES_SWITCH], size_t numCSS, double cssSensorCos[MAX_N_CSS_MEAS], double sensorUseThresh, double cssNHat_B[MAX_NUM_CSS_SENSORS*3], double *obs, double *yMeas, int *numObs, double *measMat);
    
    void sunlineKalmanGain(double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double *kalmanGain);
    
    void sunlineDynMatrix(double states[EKF_N_STATES_SWITCH], double bVec[SKF_N_STATES], double dt, double *dynMat);

    void sunlineCKFUpdate(double xBar[EKF_N_STATES_SWITCH], double kalmanGain[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS], double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double *x, double *covar);
    
    void sunlineSEKFUpdate(double kalmanGain[EKF_N_STATES_SWITCH*MAX_N_CSS_MEAS], double covarBar[EKF_N_STATES_SWITCH*EKF_N_STATES_SWITCH], double qObsVal, size_t numObs, double yObs[MAX_N_CSS_MEAS], double hObs[MAX_N_CSS_MEAS*EKF_N_STATES_SWITCH], double *states, double *x, double *covar);
    
    void sunlineSEKFSwitch(double *bVec_B, double *states, double *covar);
    
    void sunlineSEKFComputeDCM_BS(double sunheading[SKF_N_STATES_HALF], double bVec[SKF_N_STATES_HALF], double *dcm);
    
#ifdef __cplusplus
}
#endif


#endif
