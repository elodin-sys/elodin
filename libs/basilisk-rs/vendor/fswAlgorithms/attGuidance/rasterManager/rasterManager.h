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

#ifndef _RASTER_MANAGER_
#define _RASTER_MANAGER_


#include <stdint.h>

#include "cMsgCInterface/AttStateMsg_C.h"

#include "architecture/utilities/bskLogging.h"





#define MAX_RASTER_SET 36


/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* Declare module private variables */
    double scanningAngles[3 * MAX_RASTER_SET];      /*!< array of scanning angles */
    double scanningRates[3 * MAX_RASTER_SET];       /*!< array of scanning rates */
    double rasterTimes[MAX_RASTER_SET];             /*!< array of raster times*/
    int numRasters;                                 /*!< number of raster points */
    int scanSelector;                               /*!< scan selector variable*/
    int32_t mnvrActive;      /*!< [-] Flag indicating if we are maneuvering */
    int32_t mnvrComplete;    /*!< (-) Helpful flag indicating if the current maneuver is complete*/
    //uint64_t currentMnvrTime;
    uint64_t mnvrStartTime;                         /*!< maneuver start time */
    /* Declare module IO interfaces */
    AttStateMsg_C attStateOutMsg;                   /*!< The name of the output message containing the
                                                         commanded attitude references states  */

    /* Output attitude reference data to send */
    AttStateMsgPayload attOutSet;                   //!< output message copy
    BSKLogger *bskLogger;                             //!< BSK Logging
}rasterManagerConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rasterManager(rasterManagerConfig *configData, int64_t moduleID);
    void Reset_rasterManager(rasterManagerConfig *configData, uint64_t callTime, int64_t moduleID);
    void Update_rasterManager(rasterManagerConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
