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

#include "ephemerisUtilities.h"

double calculateChebyValue(double *chebyCoeff, uint32_t nCoeff,
                           double evalValue)
{
    double chebyPrev;
    double chebyNow;
    double chebyLocalPrev;
    double valueMult;
    double estValue;
    uint32_t i;
    
    chebyPrev = 1.0;
    chebyNow = evalValue;
    valueMult = 2.0*evalValue;
    
    estValue = chebyCoeff[0]*chebyPrev;
    if(nCoeff <= 1)
    {
        return(estValue);
    }
    estValue += chebyCoeff[1]*chebyNow;
    for(i=2; i<nCoeff; i=i+1)
    {
        chebyLocalPrev = chebyNow;
        chebyNow = valueMult*chebyNow - chebyPrev;
        chebyPrev = chebyLocalPrev;
        estValue += chebyCoeff[i]*chebyNow;
    }
    
    return(estValue);
}
