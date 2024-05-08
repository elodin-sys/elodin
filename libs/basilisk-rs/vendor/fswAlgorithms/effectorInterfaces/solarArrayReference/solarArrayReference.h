/*
 ISC License

 Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef _SOLAR_ARRAY_REFERENCE_
#define _SOLAR_ARRAY_REFERENCE_

#include <stdint.h>
#include "architecture/utilities/bskLogging.h"
#include "cMsgCInterface/NavAttMsg_C.h"
#include "cMsgCInterface/AttRefMsg_C.h"
#include "cMsgCInterface/HingedRigidBodyMsg_C.h"


enum attitudeFrame{
    referenceFrame = 0,
    bodyFrame = 1
};

/*! @brief Top level structure for the sub-module routines. */
typedef struct {

    /* declare these user-defined quantities */
    double a1Hat_B[3];              //!< solar array drive axis in body frame coordinates
    double a2Hat_B[3];              //!< solar array surface normal at zero rotation
    int attitudeFrame;              //!< flag = 1: compute theta reference based on current attitude instead of attitude reference

    /* declare these variables for internal computations */
    int                 count;                    //!< counter variable for finite differences
    uint64_t            priorT;                   //!< prior call time for finite differences
    double              priorThetaR;              //!< prior output msg for finite differences

    /* declare module IO interfaces */
    NavAttMsg_C            attNavInMsg;                  //!< input msg measured attitude
    AttRefMsg_C            attRefInMsg;                  //!< input attitude reference message
    HingedRigidBodyMsg_C   hingedRigidBodyInMsg;         //!< input hinged rigid body message
    HingedRigidBodyMsg_C   hingedRigidBodyRefOutMsg;     //!< output msg containing hinged rigid body target angle and angle rate

    BSKLogger *bskLogger;                         //!< BSK Logging

}solarArrayReferenceConfig;

#ifdef __cplusplus
extern "C" {
#endif

    void SelfInit_solarArrayReference(solarArrayReferenceConfig *configData, int64_t moduleID);
    void Reset_solarArrayReference(solarArrayReferenceConfig *configData, uint64_t callTime, int64_t moduleID);
    void Update_solarArrayReference(solarArrayReferenceConfig *configData, uint64_t callTime, int64_t moduleID);

#ifdef __cplusplus
}
#endif


#endif
