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

#ifndef _MRP_FEEDBACK_CONTROL_H_
#define _MRP_FEEDBACK_CONTROL_H_

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>

#include "cMsgCInterface/RWSpeedMsg_C.h"
#include "cMsgCInterface/RWAvailabilityMsg_C.h"
#include "cMsgCInterface/RWArrayConfigMsg_C.h"
#include "cMsgCInterface/VehicleConfigMsg_C.h"
#include "cMsgCInterface/AttGuidMsg_C.h"
#include "cMsgCInterface/CmdTorqueBodyMsg_C.h"


/*! @brief Data configuration structure for the MRP feedback attitude control routine. */
typedef struct {
    double K;                           //!< [rad/sec] Proportional gain applied to MRP errors
    double P;                           //!< [N*m*s]   Rate error feedback gain applied
    double Ki;                          //!< [N*m]     Integration feedback error on rate error
    double integralLimit;               //!< [N*m]     Integration limit to avoid wind-up issue
    int    controlLawType;              //!<           Flag to choose between the two control laws available
    uint64_t priorTime;                 //!< [ns]      Last time the attitude control is called
    double z[3];                        //!< [rad]     integral state of delta_omega
    double int_sigma[3];                //!< [s]       integral of the MPR attitude error
    double knownTorquePntB_B[3];        //!< [N*m]     known external torque in body frame vector components

    double ISCPntB_B[9];                //!< [kg m^2]  Spacecraft Inertia
    RWArrayConfigMsgPayload rwConfigParams; //!< [-] struct to store message containing RW config parameters in body B frame

    /* declare module IO interfaces */
    RWSpeedMsg_C rwSpeedsInMsg;                         //!< RW speed input message (Optional)
    RWAvailabilityMsg_C rwAvailInMsg;                   //!< RW availability input message (Optional)
    RWArrayConfigMsg_C rwParamsInMsg;                   //!< RW parameter input message.  (Optional)
    CmdTorqueBodyMsg_C cmdTorqueOutMsg;                 //!< commanded spacecraft external control torque output message
    CmdTorqueBodyMsg_C intFeedbackTorqueOutMsg;         //!< commanded integral feedback control torque output message
    AttGuidMsg_C guidInMsg;                             //!< attitude guidance input message
    VehicleConfigMsg_C vehConfigInMsg;                  //!< vehicle configuration input message

    BSKLogger *bskLogger;                               //!< BSK Logging
}mrpFeedbackConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_mrpFeedback(mrpFeedbackConfig *configData, int64_t moduleID);
    void Update_mrpFeedback(mrpFeedbackConfig *configData, uint64_t callTime, int64_t moduleID);
    void Reset_mrpFeedback(mrpFeedbackConfig *configData, uint64_t callTime, int64_t moduleID);


#ifdef __cplusplus
}
#endif


#endif
