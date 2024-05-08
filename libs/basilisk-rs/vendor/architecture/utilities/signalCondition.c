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

#include "architecture/utilities/signalCondition.h"

#include <math.h>
#include <stdio.h>
#include <stdarg.h>

/*! This method applies the low-pass filter configuration to the newMeas that 
    is passed in.  The state is maintained in the LowPassFilterData structure
 @return void
 @param lpData The configuration data and history of the LP filter
 @param newMeas The new measurement to take in to the filter
 */
void lowPassFilterSignal(double newMeas, LowPassFilterData *lpData)
{
    /*! See documentation of algorithm in documentation for LP torque filter module*/
    double hOmeg = lpData->hStep*lpData->omegCutoff;
    lpData->currentState = (1.0/(2.0+hOmeg)*
        (lpData->currentState*(2.0-hOmeg)+hOmeg*(newMeas+lpData->currentMeas)));
    lpData->currentMeas = newMeas;
    return;
}
