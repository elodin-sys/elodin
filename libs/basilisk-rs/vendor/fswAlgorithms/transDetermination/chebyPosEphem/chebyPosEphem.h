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

#ifndef _CHEBY_POS_EPHEM_H_
#define _CHEBY_POS_EPHEM_H_

#include "cMsgCInterface/TDBVehicleClockCorrelationMsg_C.h"
#include "cMsgCInterface/EphemerisMsg_C.h"

#include "architecture/utilities/bskLogging.h"

#define MAX_CHEB_COEFF 40
#define MAX_CHEB_RECORDS 4


/*! @brief Structure that defines the layout of an Ephemeris "record."  This is
           basically the set of coefficients for the body x/y/z positions and
           the time factors associated with those coefficients
*/
typedef struct {
    uint32_t nChebCoeff;                       /*!< [-] Number chebyshev coefficients loaded into record*/
    double ephemTimeMid;                  /*!< [s] Ephemeris time (TDB) associated with the mid-point of the curve*/
    double ephemTimeRad;                  /*!< [s] "Radius" of time that curve is valid for (half of total range*/
    double posChebyCoeff[3*MAX_CHEB_COEFF];   /*!< [-] Set of chebyshev coefficients for position */
    double velChebyCoeff[3*MAX_CHEB_COEFF];   /*!< [-] Set of coefficients for the velocity estimate*/
}ChebyEphemRecord;

/*! @brief Top level structure for the Chebyshev position ephemeris 
           fit system. e
*/
typedef struct {
    EphemerisMsg_C posFitOutMsg; /*!< [-] output navigation message for pos/vel*/
    TDBVehicleClockCorrelationMsg_C clockCorrInMsg; /*!< clock correlation input message*/
    ChebyEphemRecord ephArray[MAX_CHEB_RECORDS]; /*!< [-] Array of Chebyshev records for ephemeris*/

    uint32_t coeffSelector;    /*!< [-] Index in the ephArray that we are currently using*/

    EphemerisMsgPayload outputState; /*!< [-] The local storage of the outgoing message data*/

    BSKLogger *bskLogger;   //!< BSK Logging
}ChebyPosEphemData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_chebyPosEphem(ChebyPosEphemData *configData, int64_t moduleID);
    void Update_chebyPosEphem(ChebyPosEphemData *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_chebyPosEphem(ChebyPosEphemData *configData, uint64_t callTime,
                             int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
