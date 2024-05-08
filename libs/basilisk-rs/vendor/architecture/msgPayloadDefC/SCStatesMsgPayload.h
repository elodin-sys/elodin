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

#ifndef SC_STATE_MESSAGE_H
#define SC_STATE_MESSAGE_H


/*! @brief This structure is used in the messaging system to communicate what the
 state of the vehicle is currently.*/
typedef struct {
    double r_BN_N[3];                 //!< m  Current position vector (inertial)
    double v_BN_N[3];                 //!< m/s Current velocity vector (inertial)
    double r_CN_N[3];                 //!< m  Current position of CoM vector (inertial)
    double v_CN_N[3];                 //!< m/s Current velocity of CoM vector (inertial)
    double sigma_BN[3];               //!< -- Current MRPs (inertial)
    double omega_BN_B[3];             //!< r/s Current angular velocity
    double omegaDot_BN_B[3];          //!< r/s/s Current angular acceleration
    double TotalAccumDVBdy[3];        //!< m/s Accumulated DV of center of mass in body frame coordinates
    double TotalAccumDV_BN_B[3];      //!< m/s Accumulated DV of body frame in body frame coordinates
    double TotalAccumDV_CN_N[3];      //!< m/s Accumulated DV of center of mass in inertial frame coordinates
    double nonConservativeAccelpntB_B[3];//!< m/s/s Current Spacecraft non-conservative body frame accel
    uint64_t MRPSwitchCount;          //!< -- Number of times that MRPs have switched
}SCStatesMsgPayload;



#endif
