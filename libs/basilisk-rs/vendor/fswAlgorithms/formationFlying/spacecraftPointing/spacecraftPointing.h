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

#ifndef _SPACECRAFTPOINTING_H_
#define _SPACECRAFTPOINTING_H_

#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/AttRefMsg_C.h"

#include "architecture/utilities/bskLogging.h"
#include <stdint.h>


/*! @brief Top level structure for the spacecraft pointing module.*/
typedef struct {
    AttRefMsg_C attReferenceOutMsg;                     /*!< The name of the output message */
    NavTransMsg_C chiefPositionInMsg;                   /*!< The name of the Input message of the chief */
    NavTransMsg_C deputyPositionInMsg;                  /*!< The name of the Input message of the deputy */

    double alignmentVector_B[3];                        /*!< Vector within the B-frame that points to antenna */
    double sigma_BA[3];                                 /*!< -- MRP of B-frame with respect to A-frame */
    double old_sigma_RN[3];                             /*!< -- MRP of previous timestep */
    double old_omega_RN_N[3];                           /*!< -- Omega of previous timestep */
    int i;                                              /*!< -- Flag used to set incorrect numerical answers to zero */
    uint64_t priorTime;                                 /*!< [ns] Last time the attitude control is called */
    AttRefMsgPayload attReferenceOutBuffer;                 //!< output msg copy
    
    BSKLogger *bskLogger;                             //!< BSK Logging
}spacecraftPointingConfig;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_spacecraftPointing(spacecraftPointingConfig *configData, int64_t moduleID);
    void Update_spacecraftPointing(spacecraftPointingConfig *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_spacecraftPointing(spacecraftPointingConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
