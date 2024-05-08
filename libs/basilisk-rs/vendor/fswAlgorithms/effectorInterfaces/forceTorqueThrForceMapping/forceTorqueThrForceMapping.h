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


#ifndef FORCETORQUETHRFORCEMAPPING_H
#define FORCETORQUETHRFORCEMAPPING_H

#include <stdint.h>
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "cMsgCInterface/CmdForceBodyMsg_C.h"
#include "cMsgCInterface/THRArrayConfigMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/THRArrayCmdForceMsg_C.h"
#include "architecture/utilities/bskLogging.h"

/*! @brief This module maps thruster forces for arbitrary forces and torques
 */
typedef struct {
    /* declare module public variables */
    double   rThruster_B[MAX_EFF_CNT][3];           //!< [m]     local copy of the thruster locations
    double   gtThruster_B[MAX_EFF_CNT][3];          //!< []      local copy of the thruster force unit direction vectors

    /* declare module private variables */
    uint32_t numThrusters;                          //!< []      The number of thrusters available on vehicle
    double CoM_B[3];                                //!< [m]     CoM of the s/c

    /* declare module IO interfaces */
    CmdTorqueBodyMsg_C cmdTorqueInMsg;  //!< (optional) vehicle control (Lr) input message
    CmdForceBodyMsg_C cmdForceInMsg;  //!< (optional) vehicle control force input message
    THRArrayConfigMsg_C thrConfigInMsg;  //!< thruster cluster configuration input message
    VehicleConfigMsg_C vehConfigInMsg;  //!< vehicle config input message
    THRArrayCmdForceMsg_C thrForceCmdOutMsg;  //!< thruster force command output message

    BSKLogger *bskLogger;  //!< BSK Logging
}forceTorqueThrForceMappingConfig;

#ifdef __cplusplus
extern "C" {
#endif
    void SelfInit_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig *configData, int64_t moduleID);
    void Update_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_forceTorqueThrForceMapping(forceTorqueThrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif

#endif
