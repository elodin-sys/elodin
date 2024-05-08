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

#include "fswAlgorithms/sensorInterfaces/CSSSensorData/cssComm.h"
#include "architecture/utilities/linearAlgebra.h"
#include <string.h>
#include <stdio.h> 

/*! This method initializes the configData for theCSS sensor interface.
 It checks to ensure that the inputs are sane and then creates the
 output message
 @return void
 @param configData The configuration data associated with the CSS sensor interface
 @param moduleID The ID associated with the configData
 */
void SelfInit_cssProcessTelem(CSSConfigData *configData, int64_t moduleID)
{
    CSSArraySensorMsg_C_init(&configData->cssArrayOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the guidance module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_cssProcessTelem(CSSConfigData *configData, uint64_t callTime, int64_t moduleID)
{
    // check if the required message has not been connected
    if (!CSSArraySensorMsg_C_isLinked(&configData->sensorListInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: cssComm.sensorListInMsg wasn't connected.");
    }

    /*! - Check to make sure that number of sensors is less than the max and warn if none are set*/
    if(configData->numSensors > MAX_NUM_CSS_SENSORS)
    {
        char info[MAX_LOGGING_LENGTH];
        sprintf(info, "The configured number of CSS sensors exceeds the maximum, %d > %d! Changing the number of sensors to the max.", configData->numSensors, MAX_NUM_CSS_SENSORS);
        _bskLog(configData->bskLogger, BSK_WARNING, info);
        configData->numSensors = MAX_NUM_CSS_SENSORS;
    }
    else if (configData->numSensors == 0)
    {
        _bskLog(configData->bskLogger, BSK_WARNING, "There are zero CSS configured!");
    }
    
    if (configData->maxSensorValue == 0)
    {
        _bskLog(configData->bskLogger, BSK_WARNING, "Max CSS sensor value configured to zero! CSS sensor values will be normalized by zero, inducing faux saturation!");
    }

    configData->inputValues = CSSArraySensorMsg_C_zeroMsgPayload();

    return;
}


/*! This method takes the raw sensor data from the coarse sun sensors and
 converts that information to the format used by the CSS nav.
 @return void
 @param configData The configuration data associated with the CSS interface
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_cssProcessTelem(CSSConfigData *configData, uint64_t callTime,
    int64_t moduleID)
{
    uint32_t i, j;
    CSSArraySensorMsgPayload inMsgBuffer;
    double inputValues[MAX_NUM_CSS_SENSORS]; /* [-] Current measured CSS value for the constellation of CSS sensor */
    double ChebyDiffFactor, ChebyPrev, ChebyNow, ChebyLocalPrev, ValueMult; /* Parameters used for the Chebyshev Recursion Forumula */
    CSSArraySensorMsgPayload outputBuffer;

    outputBuffer = CSSArraySensorMsg_C_zeroMsgPayload();

    // read sensor list input msg
    inMsgBuffer = CSSArraySensorMsg_C_read(&configData->sensorListInMsg);
    vCopy(inMsgBuffer.CosValue, MAX_NUM_CSS_SENSORS, inputValues);

    /*! - Loop over the sensors and compute data
         -# Check appropriate range on sensor and calibrate
         -# If Chebyshev polynomials are configured:
             - Seed polynominal computations
             - Loop over polynominals to compute estimated correction factor
             - Output is base value plus the correction factor
         -# If sensor output range is incorrect, set output value to zero
     */
    for(i=0; i<configData->numSensors; i++)
    {
        outputBuffer.CosValue[i] = (float) inputValues[i]/configData->maxSensorValue; /* Scale Sensor Data */
        
        /* Seed the polynomial computations */
        ValueMult = 2.0*outputBuffer.CosValue[i];
        ChebyPrev = 1.0;
        ChebyNow = outputBuffer.CosValue[i];
        ChebyDiffFactor = 0.0;
        ChebyDiffFactor = configData->chebyCount > 0 ? ChebyPrev*configData->kellyCheby[0] : ChebyDiffFactor; /* if only first order correction */
        ChebyDiffFactor = configData->chebyCount > 1 ? ChebyNow*configData->kellyCheby[1] + ChebyDiffFactor : ChebyDiffFactor; /* if higher order (> first) corrections */
        
        /* Loop over remaining polynomials and add in values */
        for(j=2; j<configData->chebyCount; j = j+1)
        {
            ChebyLocalPrev = ChebyNow;
            ChebyNow = ValueMult*ChebyNow - ChebyPrev;
            ChebyPrev = ChebyLocalPrev;
            ChebyDiffFactor += configData->kellyCheby[j]*ChebyNow;
        }
        
        outputBuffer.CosValue[i] = outputBuffer.CosValue[i] + ChebyDiffFactor;
        
        if(outputBuffer.CosValue[i] > 1.0)
        {
            outputBuffer.CosValue[i] = 1.0;
        }
        else if(outputBuffer.CosValue[i] < 0.0)
        {
            outputBuffer.CosValue[i] = 0.0;
        }
    }
    
    /*! - Write aggregate output into output message */
    CSSArraySensorMsg_C_write(&outputBuffer, &configData->cssArrayOutMsg, moduleID, callTime);
    
    return;
}
