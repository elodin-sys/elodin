/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder

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


#ifndef DIPOLEMAPPING_H
#define DIPOLEMAPPING_H

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/DipoleRequestBodyMsg_C.h"
#include "cMsgCInterface/MTBCmdMsg_C.h"
#include "cMsgCInterface/MTBArrayConfigMsg_C.h"
#include "cMsgCInterface/TAMSensorBodyMsg_C.h"

/*! @brief Top level structure for the sub-module routines. */
typedef struct {
    /* Configs.*/
    double steeringMatrix[MAX_EFF_CNT * 3];             //!< matrix for mapping body frame dipole request to individual torque bar dipoles
    
    /* Inputs. */
    MTBArrayConfigMsg_C mtbArrayConfigParamsInMsg;      //!< input message containing configuration parameters for all the torque bars on the vehicle
    DipoleRequestBodyMsg_C dipoleRequestBodyInMsg;      //!< [A-m2] input message containing the requested body frame dipole

    /* Outputs. */
    MTBCmdMsg_C dipoleRequestMtbOutMsg;                 //!< [A-m2] output message containing the individual dipole requests for each torque bar on the vehicle

    /* Other. */
    MTBArrayConfigMsgPayload mtbArrayConfigParams;      //!< configuration parameters for all the torque bars used on the vehicle
    BSKLogger *bskLogger;                               //!< BSK Logging
}dipoleMappingConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_dipoleMapping(dipoleMappingConfig *configData, int64_t moduleID);
    void Update_dipoleMapping(dipoleMappingConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_dipoleMapping(dipoleMappingConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
