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

#ifndef _IMU_SENSOR_MESSAG_H
#define _IMU_SENSOR_MESSAG_H


//! @brief Simulated IMU Sensor output message definition.
typedef struct{
    double DVFramePlatform[3];      //!< m/s Accumulated DVs in platform
    double AccelPlatform[3];        //!< m/s2 Apparent acceleration of the platform
    double DRFramePlatform[3];      //!< r  Accumulated DRs in platform
    double AngVelPlatform[3];       //!< r/s Angular velocity in platform frame
}IMUSensorMsgPayload;


#endif
