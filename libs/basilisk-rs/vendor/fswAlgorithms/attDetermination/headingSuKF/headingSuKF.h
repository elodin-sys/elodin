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

#ifndef _HEADING_UKF_H_
#define _HEADING_UKF_H_

#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/HeadingFilterMsg_C.h"
#include "cMsgCInterface/OpNavMsg_C.h"
#include "cMsgCInterface/CameraConfigMsg_C.h"

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"


/*! @brief Top level structure for the SuKF heading module data */
typedef struct {
    OpNavMsg_C opnavDataOutMsg;             /*!< output message */
    HeadingFilterMsg_C filtDataOutMsg;      /*!< output message */
    OpNavMsg_C opnavDataInMsg;              /*!< input message */
    CameraConfigMsg_C cameraConfigInMsg;    /*!< (optional) input message */
    
    int putInCameraFrame;         /*!< [-] If camera message is found output the result to the camera frame as well as the body and inertial frame*/
	int numStates;                /*!< [-] Number of states for this filter*/
	int countHalfSPs;             /*!< [-] Number of sigma points over 2 */
	int numObs;                   /*!< [-] Number of measurements this cycle */
	double beta;                  /*!< [-] Beta parameter for filter */
	double alpha;                 /*!< [-] Alpha parameter for filter*/
	double kappa;                 /*!< [-] Kappa parameter for filter*/
	double lambdaVal;             /*!< [-] Lambda parameter for filter*/
	double gamma;                 /*!< [-] Gamma parameter for filter*/
    double qObsVal;               /*!< [-] OpNav instrument noise parameter*/
    double rNorm;                 /*!< [-] OpNav measurment norm*/
	double dt;                    /*!< [s] seconds since last data epoch */
	double timeTag;               /*!< [s]  Time tag for statecovar/etc */
    double noiseSF;               /*!< [-]  Scale factor for noise */
    
    double bVec_B[HEAD_N_STATES];       /*!< [-] current vector of the b frame used to make frame */
    double switchTresh;                 /*!< [-]  Threshold for switching frames */
    
    double stateInit[HEAD_N_STATES_SWITCH];    /*!< [-] State to initialize filter to*/
    double state[HEAD_N_STATES_SWITCH];        /*!< [-] State estimate for time TimeTag*/
    
	double wM[2 * HEAD_N_STATES_SWITCH + 1]; /*!< [-] Weighting vector for sigma points*/
	double wC[2 * HEAD_N_STATES_SWITCH + 1]; /*!< [-] Weighting vector for sigma points*/

	double sBar[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];         /*!< [-] Time updated covariance */
    double covarInit[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];    /*!< [-] covariance to init to*/
	double covar[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];        /*!< [-] covariance */
    double xBar[HEAD_N_STATES_SWITCH];                              /*!< [-] Current mean state estimate*/

	double obs[OPNAV_MEAS];                                         /*!< [-] Observation vector for frame*/
	double yMeas[OPNAV_MEAS*(2*HEAD_N_STATES_SWITCH+1)];            /*!< [-] Measurement model data */
    double postFits[OPNAV_MEAS];                                    /*!< [-] PostFit residuals */
    
	double SP[(2*HEAD_N_STATES_SWITCH+1)*HEAD_N_STATES_SWITCH];     /*!< [-]    sigma point matrix */

	double qNoise[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];       /*!< [-] process noise matrix */
	double sQnoise[HEAD_N_STATES_SWITCH*HEAD_N_STATES_SWITCH];      /*!< [-] cholesky of Qnoise */

	double qObs[OPNAV_MEAS*OPNAV_MEAS];  /*!< [-] Maximally sized obs noise matrix*/
    

    double sensorUseThresh;  /*!< -- Threshold below which we discount sensors*/
	NavAttMsgPayload outputHeading;   /*!< -- Output heading estimate data */
    OpNavMsgPayload opnavInBuffer;  /*!< -- message buffer */
    
    BSKLogger *bskLogger;                             //!< BSK Logging

}HeadingSuKFConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_headingSuKF(HeadingSuKFConfig *configData, int64_t moduleID);
    void Update_headingSuKF(HeadingSuKFConfig *configData, uint64_t callTime,
        int64_t moduleID);
	void Reset_headingSuKF(HeadingSuKFConfig *configData, uint64_t callTime,
		int64_t moduleID);
	void headingSuKFTimeUpdate(HeadingSuKFConfig *configData, double updateTime);
    void headingSuKFMeasUpdate(HeadingSuKFConfig *configData, double updateTime);
	void headingStateProp(double *stateInOut,  double *b_vec, double dt);
    void headingSuKFMeasModel(HeadingSuKFConfig *configData);
    void headingSuKFComputeDCM_BS(double heading[HEAD_N_STATES], double bVec[HEAD_N_STATES], double *dcm);
    void headingSuKFSwitch(double *bVec_B, double *states, double *covar);

#ifdef __cplusplus
}
#endif


#endif
