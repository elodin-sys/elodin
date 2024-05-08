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

#ifndef _GaussMarkov_HH_
#define _GaussMarkov_HH_

#include <string>
#include <stdint.h>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "architecture/utilities/bskLogging.h"


/*! @brief This module is used to apply a second-order bounded Gauss-Markov random walk
    on top of an upper level process.  The intent is that the caller will perform
    the set methods (setUpperBounds, setNoiseMatrix, setPropMatrix) as often as
    they need to, call computeNextState, and then call getCurrentState cyclically
*/
class GaussMarkov
{
    
public:
    GaussMarkov();
    GaussMarkov(uint64_t size, uint64_t newSeed = 0x1badcad1); //!< class constructor
    ~GaussMarkov();
    void computeNextState();

    /*!@brief Method does just what it says, seeds the random number generator
       @param newSeed The seed to use in the random number generator
       @return void*/
    void setRNGSeed(uint64_t newSeed) {rGen.seed((unsigned int)newSeed); RNGSeed = newSeed;}

    /*!@brief Method returns the current random walk state from model
       @return The private currentState which is the vector of random walk values*/
    Eigen::VectorXd getCurrentState() {return(currentState);}

    /*!@brief Set the upper bounds on the random walk to newBounds
       @param newBounds the bounds to put on the random walk states
       @return void*/
    void setUpperBounds(Eigen::VectorXd newBounds){stateBounds = newBounds;}

    /*!@brief Set the noiseMatrix that is used to define error sigmas
       @param noise The new value to use for the noiseMatrix variable (error sigmas)
       @return void*/
    void setNoiseMatrix(Eigen::MatrixXd noise){noiseMatrix = noise;}

    /*!@brief Set the propagation matrix that is used to propagate the state.
       @param prop The new value for the state propagation matrix
       @return void*/
    void setPropMatrix(Eigen::MatrixXd prop){propMatrix = prop;}

    Eigen::VectorXd stateBounds;  //!< -- Upper bounds to use for markov
    Eigen::VectorXd currentState;  //!< -- State of the markov model
    Eigen::MatrixXd propMatrix;    //!< -- Matrix to propagate error state with
    Eigen::MatrixXd noiseMatrix;   //!< -- Cholesky-decomposition or matrix square root of the covariance matrix to apply errors with
    BSKLogger bskLogger;                      //!< -- BSK Logging

private:
    uint64_t RNGSeed;                 //!< -- Seed for random number generator
    std::minstd_rand rGen; //!< -- Random number generator for model
    std::normal_distribution<double> rNum;  //!< -- Random number distribution for model
    uint64_t numStates;             //!< -- Number of states to generate noise for
};


#endif /* _GaussMarkov_HH_ */
