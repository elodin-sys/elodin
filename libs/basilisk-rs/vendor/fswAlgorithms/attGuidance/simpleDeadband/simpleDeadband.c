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
    Attitude Tracking simple Module
 
 */

/* modify the path to reflect the new module names */
#include "fswAlgorithms/attGuidance/simpleDeadband/simpleDeadband.h"
#include <string.h>
#include <math.h>
#include "fswAlgorithms/fswUtilities/fswDefinitions.h"
#include "architecture/utilities/macroDefinitions.h"

/* update this include to reflect the required module input messages */
#include "fswAlgorithms/attGuidance/attTrackingError/attTrackingError.h"



/*
 Pull in support files from other modules.  Be sure to use the absolute path relative to Basilisk directory.
 */
#include "architecture/utilities/linearAlgebra.h"


/*! This method initializes the configData for this module.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The ID associated with the configData
 */
void SelfInit_simpleDeadband(simpleDeadbandConfig *configData, int64_t moduleID)
{
    AttGuidMsg_C_init(&configData->attGuidOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the MRP steering control
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_simpleDeadband(simpleDeadbandConfig *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required input messages are included
    if (!AttGuidMsg_C_isLinked(&configData->guidInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: simpleDeadband.guidInMsg wasn't connected.");
    }
    configData->wasControlOff = 1;
}

/*! This method parses the input data, checks if the deadband needs to be applied and outputs
 the guidance command with simples either zeroed (control OFF) or left unchanged (control ON)
 @return void
 @param configData The configuration data associated with the attitude tracking simple module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_simpleDeadband(simpleDeadbandConfig *configData, uint64_t callTime, int64_t moduleID)
{
    /*! - Read the input message and set it as the output by default */
    configData->attGuidOut = AttGuidMsg_C_read(&configData->guidInMsg);

    /*! - Evaluate average simple in attitude and rates */
    configData->attError = 4.0 * atan(v3Norm(configData->attGuidOut.sigma_BR));
    configData->rateError = v3Norm(configData->attGuidOut.omega_BR_B);
    
    /*! - Check whether control should be ON or OFF */
    applyDBLogic_simpleDeadband(configData);
    
    /*! - Write output guidance message and update module knowledge of control status*/
    AttGuidMsg_C_write(&configData->attGuidOut, &configData->attGuidOutMsg, moduleID, callTime);
    return;
}


/*! This method applies a two-level deadbanding logic (according to the current average simple compared with the set threshold)
 and decides whether control should be switched ON/OFF or not.
 @return void
 @param configData The configuration data associated with the attitude tracking simple module
 */
void applyDBLogic_simpleDeadband(simpleDeadbandConfig *configData)
{
    uint32_t areErrorsBelowUpperThresh = (configData->attError < configData->outerAttThresh && configData->rateError < configData->outerRateThresh);
    uint32_t areErrorsBelowLowerThresh = (configData->attError < configData->innerAttThresh && configData->rateError < configData->innerRateThresh);
    
    if (areErrorsBelowUpperThresh)
    {
        if ((areErrorsBelowLowerThresh == 1) || ((areErrorsBelowLowerThresh == 0) && configData->wasControlOff))
        {
            /* Set simples to zero in order to turn off control */
            v3SetZero(configData->attGuidOut.sigma_BR);
            v3SetZero(configData->attGuidOut.omega_BR_B);
            configData->wasControlOff = 1;
        } else {
            configData->wasControlOff = 0;
        }
    } else { configData->wasControlOff = 0; }
}




