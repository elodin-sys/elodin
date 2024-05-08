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

#include <iostream>
#include <math.h>
#include "gauss_markov.h"
#include "linearAlgebra.h"

/*! The constructor initialies the random number generator used for the walks*/
GaussMarkov::GaussMarkov()
{
    this->RNGSeed = 0x1badcad1;
    std::normal_distribution<double>::param_type updatePair(0.0, 1.0/3.0);
    this->rGen.seed((unsigned int)this->RNGSeed);
    this->rNum.param(updatePair);
}

GaussMarkov::GaussMarkov(uint64_t size, uint64_t newSeed) : GaussMarkov() {
    this->RNGSeed = newSeed;
    this->propMatrix.resize(size,size);
    this->propMatrix.fill(0.0);
    this->currentState.resize((int64_t) size);
    this->currentState.fill(0.0);
    this->noiseMatrix.resize((int64_t) size, (int64_t) size);
    this->noiseMatrix.fill(0.0);
    this->stateBounds.resize((int64_t) size);
    this->stateBounds.fill(0.0);
    this->numStates = size;
    this->rGen.seed((unsigned int)this->RNGSeed);
}

/*! The destructor is a placeholder for one that might do something*/
GaussMarkov::~GaussMarkov()
{
}

/*! This method performs almost all of the work for the Gauss Markov random 
    walk.  It uses the current random walk configuration, propagates the current 
    state, and then applies appropriate errors to the states to set the current 
    error level.
    @return void
*/
void GaussMarkov::computeNextState()
{
    Eigen::VectorXd errorVector;
    Eigen::VectorXd ranNums;
    size_t i;
    
    //! - Check for consistent sizes on all of the user-settable matrices.  Quit if they don't match.
    if((this->propMatrix.size() != this->noiseMatrix.size()) ||
       ((uint64_t) this->propMatrix.size() != this->numStates*this->numStates))
    {
        bskLogger.bskLog(BSK_ERROR, "For the Gauss Markov model, you HAVE, and I mean HAVE, to have your propagate and noise matrices be same size and that size is your number of states squared.  I quit.");
        return;
    }
    if( (uint64_t) this->stateBounds.size() != this->numStates)
    {
        bskLogger.bskLog(BSK_ERROR, "For the Gauss Markov model, you HAVE, and I mean HAVE, to have your walk bounds length equal to your number of states. I quit.");
        return;
    }

    //! - Propagate the state forward in time using the propMatrix and the currentState
    errorVector = this->currentState;
    this->currentState = this->propMatrix * errorVector;
    
    //! - Compute the random numbers used for each state.  Note that the same generator is used for all
    ranNums.resize((int64_t) this->numStates);

    for(i = 0; i<this->numStates; i++)
    {
        ranNums[i] = this->rNum(rGen);
        if (this->stateBounds[i] > 0.0){
            
            double stateCalc = fabs(this->currentState[i]) > this->stateBounds[i]*1E-10 ? fabs(this->currentState[i]) : this->stateBounds[i];
            
            double boundCheck = (this->stateBounds[i]*2.0 - stateCalc)/stateCalc;
            boundCheck = boundCheck > this->stateBounds[i]*1E-10 ? boundCheck : this->stateBounds[i]*1E-10;
            boundCheck = 1.0/exp(boundCheck*boundCheck*boundCheck);
            boundCheck *= copysign(boundCheck, -this->currentState[i]);
            ranNums[i] += boundCheck;
        }
    }

    //! - Apply the noise matrix to the random numbers to get error values
    errorVector = this->noiseMatrix * ranNums;

    //! - Add the new errors to the currentState to get a good currentState
    this->currentState += errorVector;

}

