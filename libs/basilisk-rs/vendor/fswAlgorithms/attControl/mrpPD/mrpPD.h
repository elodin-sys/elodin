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

#ifndef _MRP_PD_CONTROL_H_
#define _MRP_PD_CONTROL_H_

#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"
#include "architecture/utilities/bskLogging.h"
#include <stdint.h>



/*! @brief Module configuration message definition. */
typedef struct {
    /* declare public module variables */
    double K;                           //!< [rad/sec] Proportional gain applied to MRP errors
    double P;                           //!< [N*m*s]   Rate error feedback gain applied
    double knownTorquePntB_B[3];        //!< [N*m]     known external torque in body frame vector components

    /* declare private module variables */
    double ISCPntB_B[9];                //!< [kg m^2] Spacecraft Inertia

    /* declare module IO interfaces */
    CmdTorqueBodyMsg_C cmdTorqueOutMsg;                 //!< commanded torque output message
    AttGuidMsg_C guidInMsg;                             //!< attitude guidance input message
    VehicleConfigMsg_C vehConfigInMsg;                  //!< vehicle configuration input message

    BSKLogger *bskLogger;                               //!< BSK Logging

}MrpPDConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_mrpPD(MrpPDConfig *configData, int64_t moduleID);
    void Update_mrpPD(MrpPDConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_mrpPD(MrpPDConfig *configData, uint64_t callTime, int64_t moduleID);
    
#ifdef __cplusplus
}
#endif


#endif
