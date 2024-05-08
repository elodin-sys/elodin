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

#include "fswAlgorithms/transDetermination/oeStateEphem/oeStateEphem.h"
#include "fswAlgorithms/transDetermination/_GeneralModuleFiles/ephemerisUtilities.h"
#include "fswAlgorithms/transDetermination/chebyPosEphem/chebyPosEphem.h"
#include "architecture/utilities/macroDefinitions.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/orbitalMotion.h"
#include "architecture/utilities/astroConstants.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*! This method creates the output navigation message (translation only) for 
    the ephemeris model
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param moduleID The module identification integer
 */
void SelfInit_oeStateEphem(OEStateEphemData *configData, int64_t moduleID)
{
    EphemerisMsg_C_init(&configData->stateFitOutMsg);
}


/*! This Reset method is empty
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Reset_oeStateEphem(OEStateEphemData *configData, uint64_t callTime,
                         int64_t moduleID)
{
    // check if the required message has not been connected
    if (!TDBVehicleClockCorrelationMsg_C_isLinked(&configData->clockCorrInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: oeStateEphem.clockCorrInMsg wasn't connected.");
    }
}

/*! This method takes the current time and computes the state of the object
    using that time and the stored Chebyshev coefficients.  If the time provided 
    is outside the specified range, the position vectors rail high/low appropriately.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Update_oeStateEphem(OEStateEphemData *configData, uint64_t callTime, int64_t moduleID)
{
    double currentScaledValue;              /* [s] scaled time value to within [-1,1] */
    double currentEphTime;                  /* [s] current ephemeris time */
    double smallestTimeDifference;          /* [s] smallest difference to the time interval mid-point */
    double timeDifference;                  /* [s] time difference with respect to an interval mid-point */
    double anomalyAngle;                    /* [r] general anomaly angle variable */
    ChebyOERecord *currRec;                 /* []  pointer to the current Chebyshev record being used */
    int i;
    TDBVehicleClockCorrelationMsgPayload localCorr;
    EphemerisMsgPayload tmpOutputState;
    classicElements orbEl;

    tmpOutputState = EphemerisMsg_C_zeroMsgPayload();

    /*! - read in the input message */
    localCorr = TDBVehicleClockCorrelationMsg_C_read(&configData->clockCorrInMsg);

    /*! - compute time for fitting interval */
    currentEphTime = callTime*NANO2SEC;
    currentEphTime += localCorr.ephemerisTime - localCorr.vehicleClockTime;

    /*! - select the fitting coefficients for the nearest fit interval */
    configData->coeffSelector = 0;
    smallestTimeDifference = fabs(currentEphTime - configData->ephArray[0].ephemTimeMid);
    for(i=1; i<MAX_OE_RECORDS; i++)
    {
        timeDifference = fabs(currentEphTime - configData->ephArray[i].ephemTimeMid);
        if(timeDifference < smallestTimeDifference)
        {
            configData->coeffSelector = (uint32_t) i;
            smallestTimeDifference = timeDifference;
        }
    }

    /*! - determine the scaled fitting time */
    currRec = &(configData->ephArray[configData->coeffSelector]);
    currentScaledValue = (currentEphTime - currRec->ephemTimeMid)/currRec->ephemTimeRad;
    if(fabs(currentScaledValue) > 1.0)
    {
        currentScaledValue = currentScaledValue/fabs(currentScaledValue);
    }

    /* - determine orbit elements from chebychev polynominals */
    tmpOutputState.timeTag = callTime*NANO2SEC;
    orbEl.rPeriap = calculateChebyValue(currRec->rPeriapCoeff, currRec->nChebCoeff,
                                  currentScaledValue);
    orbEl.i = calculateChebyValue(currRec->incCoeff, currRec->nChebCoeff,
                                  currentScaledValue);
    orbEl.e = calculateChebyValue(currRec->eccCoeff, currRec->nChebCoeff,
                                  currentScaledValue);
    orbEl.omega = calculateChebyValue(currRec->argPerCoeff, currRec->nChebCoeff,
                                  currentScaledValue);
    orbEl.Omega = calculateChebyValue(currRec->RAANCoeff, currRec->nChebCoeff,
                                  currentScaledValue);
    anomalyAngle = calculateChebyValue(currRec->anomCoeff, currRec->nChebCoeff,
                                   currentScaledValue);

    /*! - determine the true anomaly angle */
    if (currRec->anomalyFlag == 0) {
        orbEl.f = anomalyAngle;
    } else if (orbEl.e < 1.0) {
        /* input is mean elliptic anomaly angle */
        orbEl.f = E2f(M2E(anomalyAngle, orbEl.e), orbEl.e);
    } else {
        /* input is mean hyperbolic anomaly angle */
        orbEl.f = H2f(N2H(anomalyAngle, orbEl.e), orbEl.e);
    }

    /*! - determine semi-major axis */
    if (fabs(orbEl.e - 1.0) > 1e-12) {
        /* elliptic or hyperbolic case */
        orbEl.a = orbEl.rPeriap/(1.0-orbEl.e);
    } else {
        /* parabolic case, the elem2rv() function assumes a parabola has a = 0 */
        orbEl.a = 0.0;
    }

    /*! - Determine position and velocity vectors */
    elem2rv(configData->muCentral, &orbEl, tmpOutputState.r_BdyZero_N,
            tmpOutputState.v_BdyZero_N);

    /*! - Write the output message */
    EphemerisMsg_C_write(&tmpOutputState, &configData->stateFitOutMsg, moduleID, callTime);

    return;
}
