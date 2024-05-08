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

#ifndef _RELOD_UKF_H_
#define _RELOD_UKF_H_

#include <stdint.h>

#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/OpNavMsg_C.h"
#include "cMsgCInterface/OpNavFilterMsg_C.h"

#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/bskLogging.h"
#include "architecture/utilities/astroConstants.h"



/*! @brief Top level structure for the relative OD unscented kalman filter.
 Used to estimate the spacecraft's inertial position relative to a body.
 */
typedef struct {
    NavTransMsg_C navStateOutMsg;       //!< navigation output message
    OpNavFilterMsg_C filtDataOutMsg;    //!< output filter data message
    OpNavMsg_C opNavInMsg;              //!<  opnav input message
    
    size_t numStates;             //!< [-] Number of states for this filter
    size_t countHalfSPs;          //!< [-] Number of sigma points over 2
    size_t numObs;                //!< [-] Number of measurements this cycle
    double beta;                  //!< [-] Beta parameter for filter
    double alpha;                 //!< [-] Alpha parameter for filter
    double kappa;                 //!< [-] Kappa parameter for filter
    double lambdaVal;             //!< [-] Lambda parameter for filter
    double gamma;                 //!< [-] Gamma parameter for filter
    double switchMag;             //!< [-] Threshold for where we switch MRP set
    
    double dt;                     //!< [s] seconds since last data epoch
    double timeTag;                //!< [s]  Time tag for statecovar/etc
    double gyrAggTimeTag;          //!< [s] Time-tag for aggregated gyro data
    double aggSigma_b2b1[3];       //!< [-] Aggregated attitude motion from gyros
    double dcm_BdyGyrpltf[3][3];   //!< [-] DCM for converting gyro data to body frame
    double wM[2 * ODUKF_N_STATES + 1]; //!< [-] Weighting vector for sigma points
    double wC[2 * ODUKF_N_STATES + 1]; //!< [-] Weighting vector for sigma points
    
    double stateInit[ODUKF_N_STATES];    //!< [-] State estimate to initialize filter to
    double state[ODUKF_N_STATES];        //!< [-] State estimate for time TimeTag
    double statePrev[ODUKF_N_STATES];        //!< [-] State estimate for time TimeTag at previous time
    double sBar[ODUKF_N_STATES*ODUKF_N_STATES];         //!< [-] Time updated covariance
    double sBarPrev[ODUKF_N_STATES*ODUKF_N_STATES];     //!< [-] Time updated covariance at previous time
    double covar[ODUKF_N_STATES*ODUKF_N_STATES];        //!< [-] covariance
    double covarPrev[ODUKF_N_STATES*ODUKF_N_STATES];    //!< [-] covariance at previous time
    double covarInit[ODUKF_N_STATES*ODUKF_N_STATES];    //!< [-] Covariance to init filter with
    double xBar[ODUKF_N_STATES];            //!< [-] Current mean state estimate
    
    double obs[3];                               //!< [-] Observation vector for frame
    double yMeas[3*(2*ODUKF_N_STATES+1)];        //!< [-] Measurement model data
    
    double SP[(2*ODUKF_N_STATES+1)*ODUKF_N_STATES];          //!< [-]    sigma point matrix
    
    double qNoise[ODUKF_N_STATES*ODUKF_N_STATES];       //!< [-] process noise matrix
    double sQnoise[ODUKF_N_STATES*ODUKF_N_STATES];      //!< [-] cholesky of Qnoise
    double measNoise[ODUKF_N_MEAS*ODUKF_N_MEAS];      //!< [-] Measurement Noise
    double noiseSF;       //!< [-] Scale factor for Measurement Noise

    int planetIdInit;                    //!< [-] Planet being navigated inital value
    int planetId;                   //!< [-] Planet being navigated as per measurement
    uint32_t firstPassComplete;         //!< [-] Flag to know if first filter update
    double postFits[3];      //!< [-] PostFit residuals
    double timeTagOut;       //!< [s] Output time-tag information
    double maxTimeJump;      //!< [s] Maximum time jump to allow in propagation
    
    OpNavMsgPayload opNavInBuffer; //!< [-] ST sensor data read in from message bus

    BSKLogger *bskLogger;   //!< BSK Logging

}RelODuKFConfig;


#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_relODuKF(RelODuKFConfig *configData, int64_t moduleId);
    void Update_relODuKF(RelODuKFConfig *configData, uint64_t callTime,
                            int64_t moduleId);
    void Reset_relODuKF(RelODuKFConfig *configData, uint64_t callTime,
                           int64_t moduleId);
    void relODuKFTwoBodyDyn(double state[ODUKF_N_STATES], double mu, double *stateDeriv);
    int relODuKFTimeUpdate(RelODuKFConfig *configData, double updateTime);
    int relODuKFMeasUpdate(RelODuKFConfig *configData);
    void relODuKFCleanUpdate(RelODuKFConfig *configData);
    void relODStateProp(RelODuKFConfig *configData, double *stateInOut, double dt);
    void relODuKFMeasModel(RelODuKFConfig *configData);
    
#ifdef __cplusplus
}
#endif


#endif
