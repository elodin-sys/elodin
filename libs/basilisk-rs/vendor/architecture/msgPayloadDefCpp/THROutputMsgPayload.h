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

#ifndef SIM_THRUSTER_OUTPUT_MSG_H
#define SIM_THRUSTER_OUTPUT_MSG_H



/*! This structure is used in the messaging system to communicate what the
 state of the vehicle is currently.*/
typedef struct
//@cond DOXYGEN_IGNORE
THROutputMsgPayload
//@endcond
{
    double maxThrust;                    //!< N  Steady state thrust of thruster
    double thrustFactor;                 //!< -- Current Thrust Percentage
    double thrustForce = 0;              //!< N Thrust force magnitude
    double thrustForce_B[3] = {0};       //!< N  Thrust force vector in body frame components
    double thrustTorquePntB_B[3] = {0};  //!< N-m Thrust torque about point B in body frame components
    double thrusterLocation[3] = {0};    //!< m  Current position vector (inertial)
    double thrusterDirection[3] = {0};   //!< -- Unit vector of thruster pointing
}THROutputMsgPayload;


#endif
