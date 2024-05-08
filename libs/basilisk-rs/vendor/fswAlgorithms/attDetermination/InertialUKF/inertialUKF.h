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

#ifndef _INERTIAL_UKF_H_
#define _INERTIAL_UKF_H_

#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/InertialFilterMsg_C.h"
#include "cMsgCInterface/STAttMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/AccDataMsg_C.h"

#include "architecture/utilities/signalCondition.h"
#include "architecture/utilities/bskLogging.h"
#include <stdint.h>
#include <string.h>



/*! @brief Star Tracker (ST) sensor container structure.  Contains the msg input name and Id and sensor noise value.
 */
typedef struct {
    STAttMsg_C stInMsg;                       //!< star tracker input message 
    double noise[3*3];                        //!< [-] Per axis noise on the ST
}STMessage;

/*! @brief Structure to gather the ST messages and content */
typedef struct {
    int numST;                                  //!< Number of Star Trackers
    STMessage STMessages[MAX_ST_VEH_COUNT];     //!< [-] Decoded MIRU data for both camera heads
}STDataParsing;

/*! @brief Top level structure for the Inertial unscented kalman filter.
 Used to estimate the spacecraft's inertial attitude. Measurements are StarTracker data and gyro data.
 */
typedef struct {
    NavAttMsg_C navStateOutMsg;                     //!< The name of the output message
    InertialFilterMsg_C filtDataOutMsg;             //!< The name of the output filter data message
    VehicleConfigMsg_C massPropsInMsg;              //!< [-] The name of the mass props message
    RWArrayConfigMsg_C rwParamsInMsg;               //!< The name of the RWConfigParams input message
    RWSpeedMsg_C rwSpeedsInMsg;                     //!< [-] The name of the input RW speeds message
    AccDataMsg_C gyrBuffInMsg;                      //!< [-] Input message buffer from MIRU
    

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
	double wM[2 * AKF_N_STATES + 1]; //!< [-] Weighting vector for sigma points
	double wC[2 * AKF_N_STATES + 1]; //!< [-] Weighting vector for sigma points

    double stateInit[AKF_N_STATES];    //!< [-] State estimate to initialize filter to
    double state[AKF_N_STATES];        //!< [-] State estimate for time TimeTag
    double statePrev[AKF_N_STATES];        //!< [-] State estimate for time TimeTag at previous time
    double sBar[AKF_N_STATES*AKF_N_STATES];         //!< [-] Time updated covariance
    double sBarPrev[AKF_N_STATES*AKF_N_STATES];     //!< [-] Time updated covariance at previous time
    double covar[AKF_N_STATES*AKF_N_STATES];        //!< [-] covariance
    double covarPrev[AKF_N_STATES*AKF_N_STATES];    //!< [-] covariance at previous time
    double covarInit[AKF_N_STATES*AKF_N_STATES];    //!< [-] Covariance to init filter with
    double xBar[AKF_N_STATES];            //!< [-] Current mean state estimate

	double obs[3];          //!< [-] Observation vector for frame
	double yMeas[3*(2*AKF_N_STATES+1)];        //!< [-] Measurement model data

	double SP[(2*AKF_N_STATES+1)*AKF_N_STATES];          //!< [-]    sigma point matrix

	double qNoise[MAX_ST_VEH_COUNT*AKF_N_STATES*AKF_N_STATES];       //!< [-] process noise matrix
	double sQnoise[MAX_ST_VEH_COUNT*AKF_N_STATES*AKF_N_STATES];      //!< [-] cholesky of Qnoise

    double IInv[3][3];       //!< [(kg m^2)^-1] inverse of inertia tensor

    uint32_t numUsedGyros;   //!< -- Number of currently active CSS sensors
    uint32_t firstPassComplete; //!< flag
    double sigma_BNOut[3];   //!< [-] Output MRP
    double omega_BN_BOut[3]; //!< [r/s] Body rate output data
    double timeTagOut;       //!< [s] Output time-tag information
    double maxTimeJump;      //!< [s] Maximum time jump to allow in propagation

    STAttMsgPayload stSensorIn[MAX_ST_VEH_COUNT]; //!< [-] ST sensor data read in from message bus
    int stSensorOrder[MAX_ST_VEH_COUNT];    //!< [-] ST sensor data read in from message bus
    uint64_t ClockTimeST[MAX_ST_VEH_COUNT]; //!< [-] All of the ClockTimes for the STs
    int isFreshST[MAX_ST_VEH_COUNT]; //!< [-] isWritten flag for STs
    RWArrayConfigMsgPayload rwConfigParams; //!< [-] struct to store message containing RW config parameters in body B frame
    RWSpeedMsgPayload rwSpeeds;             //!< [-] Local reaction wheel speeds
    RWSpeedMsgPayload rwSpeedPrev;          //!< [-] Local reaction wheel speeds
    double speedDt;                         //!< [s] The time difference between speeds
    uint64_t timeWheelPrev;                 //!< [ns] Previous wheel time-tag from msg
    VehicleConfigMsgPayload localConfigData;   //!< [-] Vehicle configuration data
    LowPassFilterData gyroFilt[3];          //!< [-] Low-pass filters for input gyro data

    STDataParsing STDatasStruct;  //!< [-] Id of the input message buffer

    BSKLogger *bskLogger;   //!< BSK Logging
}InertialUKFConfig;


#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_inertialUKF(InertialUKFConfig *configData, int64_t moduleId);
    void Read_STMessages(InertialUKFConfig *configData);
    void Update_inertialUKF(InertialUKFConfig *configData, uint64_t callTime,
        int64_t moduleId);
	void Reset_inertialUKF(InertialUKFConfig *configData, uint64_t callTime,
		int64_t moduleId);
    void inertialUKFAggGyrData(InertialUKFConfig *configData, double prevTime,
                          double propTime, AccDataMsgPayload *gyrData);
	int inertialUKFTimeUpdate(InertialUKFConfig *configData, double updateTime);
    int inertialUKFMeasUpdate(InertialUKFConfig *configData, int currentST);
    void inertialUKFCleanUpdate(InertialUKFConfig *configData);
	void inertialStateProp(InertialUKFConfig *configData, double *stateInOut, double dt);
    void inertialUKFMeasModel(InertialUKFConfig *configData, int currentST);
    
#ifdef __cplusplus
}
#endif


#endif
