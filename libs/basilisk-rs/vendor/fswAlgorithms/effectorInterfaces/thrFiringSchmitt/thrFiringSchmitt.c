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
/*
    Thrust Firing Schmitt
 
 */

#include "fswAlgorithms/effectorInterfaces/thrFiringSchmitt/thrFiringSchmitt.h"
#include "architecture/utilities/macroDefinitions.h"
#include <stdio.h>
#include <string.h>






/*!
 \verbatim embed:rst
    This method initializes the configData for this module.  It creates a single output message of type
    :ref:`THRArrayOnTimeCmdMsgPayload`.
 \endverbatim
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The ID associated with the configData
 */
void SelfInit_thrFiringSchmitt(thrFiringSchmittConfig *configData, int64_t moduleID)
{
    THRArrayOnTimeCmdMsg_C_init(&configData->onTimeOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_thrFiringSchmitt(thrFiringSchmittConfig *configData, uint64_t callTime, int64_t moduleID)
{
	THRArrayConfigMsgPayload   localThrusterData;     /* local copy of the thruster data message */
	int 				i;

	configData->prevCallTime = 0;

	// check if the required input messages are included
	if (!THRArrayConfigMsg_C_isLinked(&configData->thrConfInMsg)) {
		_bskLog(configData->bskLogger, BSK_ERROR, "Error: thrFiringSchmitt.thrConfInMsg wasn't connected.");
	}
    if (!THRArrayCmdForceMsg_C_isLinked(&configData->thrForceInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrFiringSchmitt.thrForceInMsg wasn't connected.");
    }

	/*! - Zero and read in the support messages */
    localThrusterData = THRArrayConfigMsg_C_read(&configData->thrConfInMsg);

    /*! - store the number of installed thrusters */
	configData->numThrusters = localThrusterData.numThrusters;

    /*! - loop over all thrusters and for each copy over maximum thrust, set last state to off */
	for(i=0; i<configData->numThrusters; i++) {
		configData->maxThrust[i] = localThrusterData.thrusters[i].maxThrust;
		configData->lastThrustState[i] = BOOL_FALSE;
	}
}

/*! This method maps the input thruster command forces into thruster on times using a remainder tracking logic.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_thrFiringSchmitt(thrFiringSchmittConfig *configData, uint64_t callTime, int64_t moduleID)
{
	int 				i;
	double 				level;					/* [-] duty cycle fraction */
	double				controlPeriod;			/* [s] control period */
	double				onTime[MAX_EFF_CNT];	/* [s] array of commanded on time for thrusters */
    THRArrayCmdForceMsgPayload thrForceIn;          /* -- copy of the thruster force input message */
    THRArrayOnTimeCmdMsgPayload thrOnTimeOut;       /* -- copy of the thruster on-time output message */

    /*! - zero the output message */
    thrOnTimeOut = THRArrayOnTimeCmdMsg_C_zeroMsgPayload();

    /*! - the first time update() is called there is no information on the time step.  Here
     return either all thrusters off or on depending on the baseThrustState state */
	if(configData->prevCallTime == 0) {
		configData->prevCallTime = callTime;

		for(i = 0; i < configData->numThrusters; i++) {
			thrOnTimeOut.OnTimeRequest[i] = (double)(configData->baseThrustState) * 2.0;
		}

        THRArrayOnTimeCmdMsg_C_write(&thrOnTimeOut, &configData->onTimeOutMsg, moduleID, callTime);
		return;
	}

    /*! - compute control time period Delta_t */
	controlPeriod = ((double)(callTime - configData->prevCallTime)) * NANO2SEC;
	configData->prevCallTime = callTime;

    /*! - read the input thruster force message */
    thrForceIn = THRArrayCmdForceMsg_C_read(&configData->thrForceInMsg);

    /*! - Loop through thrusters */
	for(i = 0; i < configData->numThrusters; i++) {

        /*! - Correct for off-pulsing if necessary.  Here the requested force is negative, and the maximum thrust
         needs to be added.  If not control force is requested in off-pulsing mode, then the thruster force should
         be set to the maximum thrust value */
		if (configData->baseThrustState == 1) {
			thrForceIn.thrForce[i] += configData->maxThrust[i];
		}

        /*! - Do not allow thrust requests less than zero */
		if (thrForceIn.thrForce[i] < 0.0) {
			thrForceIn.thrForce[i] = 0.0;
		}
        /*! - Compute T_on from thrust request, max thrust, and control period */
		onTime[i] = thrForceIn.thrForce[i]/configData->maxThrust[i]*controlPeriod;

        /*! - Apply Schmitt trigger logic */
		if (onTime[i] < configData->thrMinFireTime) {
			/*! - Request is less than minimum fire time */
			level = onTime[i]/configData->thrMinFireTime;
			if (level >= configData->level_on) {
				configData->lastThrustState[i] = BOOL_TRUE;
				onTime[i] = configData->thrMinFireTime;
			} else if (level <= configData->level_off) {
				configData->lastThrustState[i] = BOOL_FALSE;
				onTime[i] = 0.0;
			} else if (configData->lastThrustState[i] == BOOL_TRUE) {
				onTime[i] = configData->thrMinFireTime;
			} else {
				onTime[i] = 0.0;
			}
		} else if (onTime[i] >= controlPeriod) {
            /*! - Request is greater than control period then oversaturate onTime */
			configData->lastThrustState[i] = BOOL_TRUE;
			onTime[i] = 1.1*controlPeriod; // oversaturate to avoid numerical error
		} else {
			/*! - Request is greater than minimum fire time and less than control period */
			configData->lastThrustState[i] = BOOL_TRUE;
		}

		/*! Set the output data */
		thrOnTimeOut.OnTimeRequest[i] = onTime[i];
	}

    THRArrayOnTimeCmdMsg_C_write(&thrOnTimeOut, &configData->onTimeOutMsg, moduleID, callTime);

	return;

}
