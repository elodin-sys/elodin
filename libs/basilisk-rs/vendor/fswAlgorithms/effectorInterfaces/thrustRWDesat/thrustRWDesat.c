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

#include "fswAlgorithms/effectorInterfaces/thrustRWDesat/thrustRWDesat.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>

/*! This method initializes the configData for the thruster-based RW desat module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the thruster desat
 @param moduleID The module ID associated with configData
 */
void SelfInit_thrustRWDesat(thrustRWDesatConfig *configData, int64_t moduleID)
{
    THRArrayOnTimeCmdMsg_C_init(&configData->thrCmdOutMsg);
}


void Reset_thrustRWDesat(thrustRWDesatConfig *configData, uint64_t callTime, int64_t moduleID)
{
    RWConstellationMsgPayload localRWData;
    THRArrayConfigMsgPayload localThrustData;
    VehicleConfigMsgPayload localConfigData;
    int i;
    double momentArm[3];
    double thrustDat_B[3];

	// check if the required input messages are included
	if (!RWConstellationMsg_C_isLinked(&configData->rwConfigInMsg)) {
		_bskLog(configData->bskLogger, BSK_ERROR, "Error: thrustRWDesat.rwConfigInMsg wasn't connected.");
	}
	if (!VehicleConfigMsg_C_isLinked(&configData->vecConfigInMsg)) {
		_bskLog(configData->bskLogger, BSK_ERROR, "Error: thrustRWDesat.vecConfigInMsg wasn't connected.");
	}
	if (!THRArrayConfigMsg_C_isLinked(&configData->thrConfigInMsg)) {
		_bskLog(configData->bskLogger, BSK_ERROR, "Error: thrustRWDesat.thrConfigInMsg wasn't connected.");
	}
    if (!RWSpeedMsg_C_isLinked(&configData->rwSpeedInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrustRWDesat.rwSpeedInMsg wasn't connected.");
    }

    /*! - Read input messages */
    localRWData = RWConstellationMsg_C_read(&configData->rwConfigInMsg);
    localConfigData = VehicleConfigMsg_C_read(&configData->vecConfigInMsg);
    localThrustData = THRArrayConfigMsg_C_read(&configData->thrConfigInMsg);

    /*! - Transform from structure S to body B frame */
    configData->numRWAs = localRWData.numRW;
    for(i=0; i<configData->numRWAs; i=i+1)
    {
        v3Copy(localRWData.reactionWheels[i].gsHat_B, &configData->rwAlignMap[i*3]);
    }

    configData->numThrusters = localThrustData.numThrusters;
    for(i=0; i<configData->numThrusters; i=i+1)
    {
        v3Copy(localThrustData.thrusters[i].tHatThrust_B, &configData->thrAlignMap[i*3]);
        v3Copy(localThrustData.thrusters[i].rThrust_B, thrustDat_B);
        v3Subtract(thrustDat_B, localConfigData.CoM_B, momentArm);
        v3Copy(localThrustData.thrusters[i].tHatThrust_B, thrustDat_B);
        v3Cross(momentArm, thrustDat_B, &(configData->thrTorqueMap[i*3]));
    }

    configData->previousFiring = 0;
    v3SetZero(configData->accumulatedImp);
    configData->totalAccumFiring = 0.0;

}

/*! This method takes in the current oberved reaction wheel angular velocities.
 @return void
 @param configData The configuration data associated with the RW desat logic
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_thrustRWDesat(thrustRWDesatConfig *configData, uint64_t callTime,
    int64_t moduleID)
{
    int32_t i;
	int32_t selectedThruster;     /* Thruster index to fire */
    RWSpeedMsgPayload rwSpeeds;   /* Local reaction wheel speeds */
	double observedSpeedVec[3];   /* The total summed speed of RWAs*/
	double singleSpeedVec[3];     /* The speed vector for a single wheel*/
	double bestMatch;             /* The current best thruster/wheel matching*/
	double currentMatch;          /* Assessment of the current match */
    double fireValue;             /* Amount of time to fire the jet for */
	THRArrayOnTimeCmdMsgPayload outputData;    /* Local output firings */
  
    /*! - If we haven't met the cooldown threshold, do nothing */
	if ((callTime - configData->previousFiring)*1.0E-9 <
		configData->thrFiringPeriod)
	{
		return;
	}

    /*! - Read the input rwheel speeds from the reaction wheels*/
    rwSpeeds = RWSpeedMsg_C_read(&configData->rwSpeedInMsg);
    
    /*! - Accumulate the total momentum vector we want to apply (subtract speed vectors)*/
	v3SetZero(observedSpeedVec);
	for (i = 0; i < configData->numRWAs; i++)
	{
		v3Scale(rwSpeeds.wheelSpeeds[i], &(configData->rwAlignMap[i * 3]), 
			singleSpeedVec);
		v3Subtract(observedSpeedVec, singleSpeedVec, observedSpeedVec);
	}

	/*! - If we are within the specified threshold for the momentum, stop desaturation.*/
	if (v3Norm(observedSpeedVec) < configData->DMThresh)
	{
		return;
	}

    /*! - Iterate through the list of thrusters and find the "best" match for the 
          observed momentum vector that does not continue to perturb the velocity 
          in the same direction as previous aggregate firings.  Only do this once we have 
		  removed the specified momentum accuracy from the current direction.*/
	selectedThruster = -1;
	bestMatch = 0.0;
	if (v3Dot(configData->currDMDir, observedSpeedVec) <= configData->DMThresh)
	{
		for (i = 0; i < configData->numThrusters; i++)
		{

			fireValue = v3Dot(observedSpeedVec,
				&(configData->thrTorqueMap[i * 3]));
			currentMatch = v3Dot(configData->accumulatedImp,
				&(configData->thrAlignMap[i * 3]));
			if (fireValue - currentMatch > bestMatch && fireValue > 0.0)
			{
				selectedThruster = i;
				bestMatch = fireValue - currentMatch;
			}
		}
		if (selectedThruster >= 0)
		{
			v3Normalize(&configData->thrTorqueMap[selectedThruster * 3],
				configData->currDMDir);
		}
	}
    
    /*! - Zero out the thruster commands prior to setting the selected thruster.
          Only apply thruster firing if the best match is non-zero.  Find the thruster 
		  that best matches the current specified direction.
    */
    outputData = THRArrayOnTimeCmdMsg_C_zeroMsgPayload();
	selectedThruster = -1;
	bestMatch = 0.0;
	for (i = 0; i < configData->numThrusters; i++)
	{

		fireValue = v3Dot(configData->currDMDir,
			&(configData->thrTorqueMap[i * 3]));
		currentMatch = v3Dot(configData->accumulatedImp,
			&(configData->thrAlignMap[i * 3]));
		if (fireValue - currentMatch > bestMatch && fireValue > 0.0)
		{
			selectedThruster = i;
			bestMatch = fireValue - currentMatch;
		}
	}
    /*! - If we have a valid match: 
          - Set firing based on the best counter to the observed momentum.
          - Saturate based on the maximum allowable firing
          - Accumulate impulse and the total firing
          - Set the previous call time value for cooldown check */
	if (bestMatch > 0.0)
	{
		outputData.OnTimeRequest[selectedThruster] = v3Dot(configData->currDMDir,
			&(configData->thrTorqueMap[selectedThruster * 3]));
		outputData.OnTimeRequest[selectedThruster] =
			outputData.OnTimeRequest[selectedThruster] > configData->maxFiring ?
			configData->maxFiring : outputData.OnTimeRequest[selectedThruster];
		configData->previousFiring = callTime;
		configData->totalAccumFiring += outputData.OnTimeRequest[selectedThruster];
        v3Scale(outputData.OnTimeRequest[selectedThruster],
                &(configData->thrAlignMap[selectedThruster * 3]), singleSpeedVec);
        v3Add(configData->accumulatedImp, singleSpeedVec,
            configData->accumulatedImp);
	}

    /*! - Write the output message to the thruster system */
    THRArrayOnTimeCmdMsg_C_write(&outputData, &configData->thrCmdOutMsg, moduleID, callTime);

    return;
}

