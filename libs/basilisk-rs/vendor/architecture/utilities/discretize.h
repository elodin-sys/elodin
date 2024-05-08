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

#ifndef _discretize_HH_
#define _discretize_HH_

#include <string>
#include <stdint.h>
#include <Eigen/Dense>


typedef enum {
    TO_ZERO,
    FROM_ZERO,
    NEAR
} roundDirection_t;

/*! This module discretizes data for output. It has the option to carry over discretization error or not.
*/
class Discretize
{
    
public:
    Discretize();
    Discretize(uint8_t numStates);
    ~Discretize();
//    void setLSBByBits(uint8_t numBits, double min, double max);
//    /*!@brief Method determines the size of an output data bin (bit-value) making sure that zero is
//     a possible output and giving proportionate numbers of bits to the size of max and min void*/

    /*!@brief Avoid calculating bit value (bin size) and just set it because a resolution is known
       @param givenLSB
       @return void*/
    void setLSB(Eigen::VectorXd givenLSB) {this->LSB = givenLSB;}

    void setRoundDirection(roundDirection_t direction);

    /*!@brief Sets the round direction (toZero, fromZero, near) for discretization
     @param carryErrorIn
     @return void*/
    void setCarryError(bool carryErrorIn){this->carryError = carryErrorIn;}

    /*!@brief Discretizes the given truth vector according to a least significant bit (binSize)
       @param undiscretizedVector
       @return vector of discretized values*/
    Eigen::VectorXd discretize(Eigen::VectorXd undiscretizedVector);

    /*!@brief Get the discretization errors
     @return the errors due to discretization in a corresponding vector*/
    Eigen::VectorXd getDiscretizationErrors(){return(this->discErrors);}

    Eigen::VectorXd LSB;                //!< -- size of bin, bit value, least significant bit
    
private:
    roundDirection_t roundDirection;    //!< -- Direction to round when discretizing. "toZero", "fromZero", and "near" are the options.
    uint8_t numStates;                  //!< -- Number of states to be discretized (length of vector fed in)
    Eigen::VectorXd discErrors;         //!< -- Errors from discretization. Can be returned to adjusted integrated values.
    bool carryError;                    //!< -- true if discError should be added next time around, false if not.
};


#endif /* _discretize_HH_ */
