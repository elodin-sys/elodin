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

#ifndef _SUNLINE_UKF_H_
#define _SUNLINE_UKF_H_

#include <stdint.h>

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/CSSArraySensorMsg_C.h"
#include "cMsgCInterface/SunlineFilterMsg_C.h"
#include "cMsgCInterface/CSSConfigMsg_C.h"

#include "architecture/utilities/bskLogging.h"




/*! @brief Top level structure for the CSS-based unscented Kalman Filter.
 Used to estimate the sun state in the vehicle body frame. */
typedef struct {
    NavAttMsg_C navStateOutMsg;                     /*!< The name of the output message*/
    SunlineFilterMsg_C filtDataOutMsg;              /*!< The name of the output filter data message*/
    CSSArraySensorMsg_C cssDataInMsg;               /*!< The name of the Input message*/
    CSSConfigMsg_C cssConfigInMsg;                  /*!< [-] The name of the CSS configuration message*/

	int numStates;                /*!< [-] Number of states for this filter*/
	int countHalfSPs;             /*!< [-] Number of sigma points over 2 */
	int numObs;                   /*!< [-] Number of measurements this cycle */
	double beta;                  /*!< [-] Beta parameter for filter */
	double alpha;                 /*!< [-] Alpha parameter for filter*/
	double kappa;                 /*!< [-] Kappa parameter for filter*/
	double lambdaVal;             /*!< [-] Lambda parameter for filter*/
	double gamma;                 /*!< [-] Gamma parameter for filter*/
    double qObsVal;               /*!< [-] CSS instrument noise parameter*/

	double dt;                     /*!< [s] seconds since last data epoch */
	double timeTag;                /*!< [s]  Time tag for statecovar/etc */

	double wM[2 * SKF_N_STATES + 1]; /*!< [-] Weighting vector for sigma points*/
	double wC[2 * SKF_N_STATES + 1]; /*!< [-] Weighting vector for sigma points*/

	double state[SKF_N_STATES];        /*!< [-] State estimate for time TimeTag*/
	double sBar[SKF_N_STATES*SKF_N_STATES];         /*!< [-] Time updated covariance */
	double covar[SKF_N_STATES*SKF_N_STATES];        /*!< [-] covariance */
    double xBar[SKF_N_STATES];            /*!< [-] Current mean state estimate*/

	double obs[MAX_N_CSS_MEAS];          /*!< [-] Observation vector for frame*/
	double yMeas[MAX_N_CSS_MEAS*(2*SKF_N_STATES+1)];        /*!< [-] Measurement model data */
    double postFits[MAX_N_CSS_MEAS];  /*!< [-] PostFit residuals */
    
	double SP[(2*SKF_N_STATES+1)*SKF_N_STATES];     /*!< [-]    sigma point matrix */

	double qNoise[SKF_N_STATES*SKF_N_STATES];       /*!< [-] process noise matrix */
	double sQnoise[SKF_N_STATES*SKF_N_STATES];      /*!< [-] cholesky of Qnoise */

	double qObs[MAX_N_CSS_MEAS*MAX_N_CSS_MEAS];  /*!< [-] Maximally sized obs noise matrix*/
    
    double cssNHat_B[MAX_NUM_CSS_SENSORS*3];     /*!< [-] CSS normal vectors converted over to body*/
    double CBias[MAX_NUM_CSS_SENSORS];       /*!< [-] CSS individual calibration coefficients */

    uint32_t numActiveCss;   /*!< -- Number of currently active CSS sensors*/
    uint32_t numCSSTotal;    /*!< [-] Count on the number of CSS we have on the spacecraft*/
    double sensorUseThresh;  /*!< -- Threshold below which we discount sensors*/
	NavAttMsgPayload outputSunline;   /*!< -- Output sunline estimate data */
    CSSArraySensorMsgPayload cssSensorInBuffer; /*!< [-] CSS sensor data read in from message bus*/

    BSKLogger *bskLogger;   //!< BSK Logging
}SunlineUKFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_sunlineUKF(SunlineUKFConfig *configData, int64_t moduleID);
    void Update_sunlineUKF(SunlineUKFConfig *configData, uint64_t callTime,
        int64_t moduleID);
	void Reset_sunlineUKF(SunlineUKFConfig *configData, uint64_t callTime,
		int64_t moduleID);
	void sunlineUKFTimeUpdate(SunlineUKFConfig *configData, double updateTime);
    void sunlineUKFMeasUpdate(SunlineUKFConfig *configData, double updateTime);
	void sunlineStateProp(double *stateInOut, double dt);
    void sunlineUKFMeasModel(SunlineUKFConfig *configData);
    
#ifdef __cplusplus
}
#endif


#endif
