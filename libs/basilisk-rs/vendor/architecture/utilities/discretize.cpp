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

#include "discretize.h"
#include <iostream>
#include <string>
#include <math.h>
#include "linearAlgebra.h"

/*! The constructor initialies the random number generator used for the walks*/
Discretize::Discretize()
{
    this->numStates = 0;
    this->roundDirection = TO_ZERO;
    this->carryError = false;
}

/*! This lets the user initialized the discretization model to the right size*/
Discretize::Discretize(uint8_t numStates) : Discretize() {
    this->numStates = numStates;
    this->LSB.resize(this->numStates);
    this->LSB.fill(0.0);
    this->discErrors.resize(this->numStates);
    this->discErrors.fill(0.0);
}

/*! The destructor is a placeholder for one that might do something*/
Discretize::~Discretize()
{
}

///*! This method calculates the least significant bit size given the maximum state value,
//    minimum state value, and number of bits to use..
//    @return void */
//void setLSBByBits(uint8_t numBits, double min, double max);

/*!@brief Sets the round direction (toZero, fromZero, near) for discretization
 @param direction
 @return void*/
void Discretize::setRoundDirection(roundDirection_t direction){
    
    this->roundDirection = direction;

    return;
}


/*!@brief Discretizes the given truth vector according to a least significant bit (binSize)
 @param undiscretizedVector
 @return vector of discretized values*/
Eigen::VectorXd Discretize::discretize(Eigen::VectorXd undiscretizedVector){
    
    if (this->carryError){
        undiscretizedVector += this->discErrors;
    }
    
    //discretize the data
    Eigen::VectorXd workingVector = undiscretizedVector.cwiseQuotient(this->LSB);
    workingVector = workingVector.cwiseAbs();
    
    if (this->roundDirection == TO_ZERO){
        for (uint8_t i = 0; i < this->numStates; i++){
            workingVector[i] = floor(workingVector[i]);
        }
    }else if(this->roundDirection == FROM_ZERO){
        for (uint8_t i = 0; i < this->numStates; i++){
            workingVector[i] = ceil(workingVector[i]);
        }
    }else if(this->roundDirection == NEAR){
        for (uint8_t i = 0; i < this->numStates; i++){
            workingVector[i] = round(workingVector[i]);
        }
    }

    workingVector = workingVector.cwiseProduct(this->LSB);
    for (uint8_t i = 0; i < this->numStates; i++){
        workingVector[i] = copysign(workingVector[i], undiscretizedVector[i]);
    }
    this->discErrors = undiscretizedVector - workingVector;
    
    return workingVector;
}


