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

#ifndef _RW_NULL_SPACE_H_
#define _RW_NULL_SPACE_H_

#include "cMsgCInterface/ArrayMotorTorqueMsg_C.h"
#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/RWConstellationMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>
#include <stdlib.h>


/*! @brief The configuration structure for the rwNullSpace module.  */
typedef struct {
    ArrayMotorTorqueMsg_C rwMotorTorqueInMsg;       //!< [-] The name of the Input message
    RWSpeedMsg_C rwSpeedsInMsg;                     //!< [-] The name of the input RW speeds
    RWSpeedMsg_C rwDesiredSpeedsInMsg;              //!< [-] (optional) The name of the desired RW speeds
    RWConstellationMsg_C rwConfigInMsg;             //!< [-] The name of the RWA configuration message
    ArrayMotorTorqueMsg_C rwMotorTorqueOutMsg;      //!< [-] The name of the output message

	double tau[MAX_EFF_CNT * MAX_EFF_CNT];          //!< [-] RW nullspace project matrix
	double OmegaGain;                               //!< [-] The gain factor applied to the RW speeds
	uint32_t numWheels;                             //!< [-] The number of reaction wheels we have

    BSKLogger *bskLogger;                             //!< BSK Logging
}rwNullSpaceConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_rwNullSpace(rwNullSpaceConfig *configData, int64_t moduleID);
    void Update_rwNullSpace(rwNullSpaceConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_rwNullSpace(rwNullSpaceConfig *configData, uint64_t callTime,
                            int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
