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

#ifndef _THRUST_GROUP_DATA_
#define _THRUST_GROUP_DATA_

#include "architecture/utilities/macroDefinitions.h"
#include "cMsgCInterface/THRArrayOnTimeCmdMsg_C.h"



/*! @brief Sub structure that contains all of the configuration data and output
    information for a single thruster group.  There can be several thruster 
    groups available in a single control scheme.
*/
typedef struct {
    double nomThrustOn;          /*!< s The nominal thruster on-time for effectors*/
    uint32_t maxNumCmds;         /*!< - The maximum number of commands to output*/
    uint32_t numEffectors;       /*!< - The number of effectors we have access to*/
    double minThrustRequest;     /*!< - The minimum allowable on-time for a thruster*/
    double thrOnMap[3*MAX_EFF_CNT]; /*!< - Mapping between on-times and torque requests*/
    THRArrayOnTimeCmdMsg_C thrOnTimeOutMsg; /*!< - The name of the output message*/
    THRArrayOnTimeCmdMsgPayload cmdRequests; /*!< - The array of on-time command requests sent to thrusters*/
}ThrustGroupData;


#endif
