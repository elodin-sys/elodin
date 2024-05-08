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

#ifndef _DV_ACCUMULATION_H_
#define _DV_ACCUMULATION_H_

#include "cMsgCInterface/NavTransMsg_C.h"
#include "cMsgCInterface/AccDataMsg_C.h"

#include "architecture/utilities/bskLogging.h"


/*! @brief Top level structure for the CSS sensor interface system.  Contains all parameters for the
 CSS interface*/
typedef struct {
    NavTransMsg_C dvAcumOutMsg; //!< accumulated DV output message
    AccDataMsg_C accPktInMsg; //!< [-] input accelerometer message
    
    uint32_t msgCount;      //!< [-] The total number of messages read from inputs
    uint32_t dvInitialized; //!< [-] Flag indicating whether DV has been started completely
    uint64_t previousTime;  //!< [ns] The clock time associated with the previous run of algorithm
    double vehAccumDV_B[3];    //!< [m/s] The accumulated Delta_V in body frame components

    BSKLogger *bskLogger;   //!< BSK Logging
}DVAccumulationData;

#ifdef __cplusplus
extern "C" {
#endif
    
    void SelfInit_dvAccumulation(DVAccumulationData *configData, int64_t moduleID);
    void Update_dvAccumulation(DVAccumulationData *configData, uint64_t callTime,
        int64_t moduleID);
    void Reset_dvAccumulation(DVAccumulationData *configData, uint64_t callTime,
                               int64_t moduleID);
    void dvAccumulation_swap(AccPktDataMsgPayload *p, AccPktDataMsgPayload *q);
    int dvAccumulation_partition(AccPktDataMsgPayload *A, int start, int end);
    void dvAccumulation_QuickSort(AccPktDataMsgPayload *A, int start, int end);
    
#ifdef __cplusplus
}
#endif


#endif
