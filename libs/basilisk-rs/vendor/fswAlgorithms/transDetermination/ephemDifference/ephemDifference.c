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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "fswAlgorithms/transDetermination/ephemDifference/ephemDifference.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"

/*! @brief This method creates the output ephemeris messages for each body.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param moduleID The module identification integer
 */
void SelfInit_ephemDifference(EphemDifferenceData *configData, int64_t moduleID)
{
    uint32_t i;
    for(i = 0; i < MAX_NUM_CHANGE_BODIES; i++)
    {
        EphemerisMsg_C_init(&configData->changeBodies[i].ephOutMsg);
    }

}


/*! @brief This method resets the module.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Reset_ephemDifference(EphemDifferenceData *configData, uint64_t callTime,
                         int64_t moduleID)
{
    // check if the required message has not been connected
    if (!EphemerisMsg_C_isLinked(&configData->ephBaseInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: ephemDifference.ephBaseInMsg wasn't connected.");
    }

    configData->ephBdyCount = 0;
    for(int i = 0; i < MAX_NUM_CHANGE_BODIES; i++)
    {
        if (EphemerisMsg_C_isLinked(&configData->changeBodies[i].ephInMsg)) {
            configData->ephBdyCount++;
        } else {
            break;
        }
    }

    if (configData->ephBdyCount == 0) {
        _bskLog(configData->bskLogger, BSK_WARNING, "Your outgoing ephemeris message count is zero. Be sure to specify desired output messages.");
    }
}

/*! @brief This method recomputes the body postions and velocities relative to
    the base body ephemeris and writes out updated ephemeris position and velocity
    for each body.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Update_ephemDifference(EphemDifferenceData *configData, uint64_t callTime, int64_t moduleID)
{
    uint32_t i;
    EphemerisMsgPayload tmpBaseEphem;
    EphemerisMsgPayload tmpEphStore;

    // read input msg
    tmpBaseEphem = EphemerisMsg_C_read(&configData->ephBaseInMsg);

    for(i = 0; i < configData->ephBdyCount; i++)
    {
        tmpEphStore = EphemerisMsg_C_read(&configData->changeBodies[i].ephInMsg);

        v3Subtract(tmpEphStore.r_BdyZero_N,
                   tmpBaseEphem.r_BdyZero_N,
                   tmpEphStore.r_BdyZero_N);
        v3Subtract(tmpEphStore.v_BdyZero_N,
                   tmpBaseEphem.v_BdyZero_N,
                   tmpEphStore.v_BdyZero_N);
        
        EphemerisMsg_C_write(&tmpEphStore, &configData->changeBodies[i].ephOutMsg, moduleID, callTime);
    }
    return;
}
