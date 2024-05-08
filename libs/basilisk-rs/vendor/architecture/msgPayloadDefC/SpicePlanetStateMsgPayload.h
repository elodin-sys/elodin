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

#ifndef SpicePlanetState_MESSAGE_H
#define SpicePlanetState_MESSAGE_H


#define MAX_BODY_NAME_LENGTH 64

//! @brief The SPICE planet statate structure is the struct used to ouput planetary body states to the messaging system
typedef struct {
    double J2000Current;            //!< s Time of validity for the planet state
    double PositionVector[3];       //!< m True position of the planet for the time
    double VelocityVector[3];       //!< m/s True velocity of the planet for the time
    double J20002Pfix[3][3];        //!< (-) Orientation matrix of planet-fixed relative to inertial
    double J20002Pfix_dot[3][3];    //!< (-) Derivative of the orientation matrix of planet-fixed relative to inertial
    int computeOrient;              //!< (-) Flag indicating whether the reference should be computed
    char PlanetName[MAX_BODY_NAME_LENGTH];        //!< -- Name of the planet for the state
}SpicePlanetStateMsgPayload;


#endif
