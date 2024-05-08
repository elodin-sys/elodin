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

#ifndef _SIGNAL_CONDITION_H_
#define _SIGNAL_CONDITION_H_


/*! struct definition */
typedef struct {
    double hStep;         /*!< [s]      filter time step (assumed to be fixed) */
    double omegCutoff;    /*!< [rad/s]  Cutoff frequency for the filter        */
    double currentState;  /*!< [-] Current state of the filter                 */
    double currentMeas;   /*!< [-] Current measurement that we read            */
}LowPassFilterData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void    lowPassFilterSignal(double newMeas, LowPassFilterData *lpData);
    
#ifdef __cplusplus
}
#endif

#endif
