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

#ifndef _Saturate_HH_
#define _Saturate_HH_

#include <stdint.h>
#include <Eigen/Dense>


/*! @brief This class is used to saturate an output variable
*/
class Saturate
{
    
public:
    Saturate();
    Saturate(int64_t size);     //!< class constructor 
    ~Saturate();
    void setBounds(Eigen::MatrixXd bounds);
    Eigen::VectorXd saturate(Eigen::VectorXd unsaturatedStates);
    /*!@brief Saturates the given unsaturated states
       @param unsaturated States, a vector of the unsaturated states
       @return saturatedStates*/
    
private:
    int64_t numStates;              //!< -- Number of states to generate noise for
    Eigen::MatrixXd stateBounds;    //!< -- one row for each state. lower bounds in left column, upper in right column
};


#endif /* _saturate_HH_ */
