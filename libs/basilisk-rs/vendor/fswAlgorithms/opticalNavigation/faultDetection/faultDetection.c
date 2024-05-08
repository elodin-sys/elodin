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

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "faultDetection.h"



/*! Self-init for the fault detection module
 @return void
 @param configData The configuration data associated with the model
 @param moduleID The module identification integer
 */
void SelfInit_faultDetection(FaultDetectionData *configData, int64_t moduleID)
{
    OpNavMsg_C_init(&configData->opNavOutMsg);
}


/*! This resets the module to original states.
 @return void
 @param configData The configuration data associated with the model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Reset_faultDetection(FaultDetectionData *configData, uint64_t callTime, int64_t moduleID)
{
    // check that the opnave messages are linked
    if (!OpNavMsg_C_isLinked(&configData->navMeasPrimaryInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: faultDetection.navMeasPrimaryInMsg wasn't connected.");
    }
    if (!OpNavMsg_C_isLinked(&configData->navMeasSecondaryInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: faultDetection.navMeasSecondaryInMsg wasn't connected.");
    }
    if (!CameraConfigMsg_C_isLinked(&configData->cameraConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: faultDetection.cameraConfigInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->attInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: faultDetection.attInMsg wasn't connected.");
    }

}

/*! This method reads in the two compared navigation messages and outputs the best measurement possible. It compares the faults of each and uses camera and attitude knowledge to output the information in all necessary frames.
 Three fault modes are possible.
 FaultMode = 0 is the less restricitve: it uses either of the measurements availabe and merges them if they are both available
 FaultMode = 1 is more restricitve: only the primary is used if both are available and the secondary is only used for a dissimilar check
 FaultMode = 2 is most restricive: the primary is not used in the abscence of the secondary measurement
 @return void
 @param configData The configuration data associated with the model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Update_faultDetection(FaultDetectionData *configData, uint64_t callTime, int64_t moduleID)
{
    OpNavMsgPayload opNavIn1;
    OpNavMsgPayload opNavIn2;
    OpNavMsgPayload opNavMsgOut;
    CameraConfigMsgPayload cameraSpecs;
    NavAttMsgPayload attInfo;
    double dcm_NC[3][3], dcm_CB[3][3], dcm_BN[3][3];

    /* zero the output message */
    opNavMsgOut = OpNavMsg_C_zeroMsgPayload();

    /*! - read input opnav messages */
    opNavIn1 = OpNavMsg_C_read(&configData->navMeasPrimaryInMsg);
    opNavIn2 = OpNavMsg_C_read(&configData->navMeasSecondaryInMsg);
    /*! - read dcm messages */
    cameraSpecs = CameraConfigMsg_C_read(&configData->cameraConfigInMsg);
    attInfo = NavAttMsg_C_read(&configData->attInMsg);
    /*! - compute dcms */
    MRP2C(cameraSpecs.sigma_CB, dcm_CB);
    MRP2C(attInfo.sigma_BN, dcm_BN);
    m33MultM33(dcm_CB, dcm_BN, dcm_NC);
    m33Transpose(dcm_NC, dcm_NC);
    
    /*! Begin fault detection logic */
    /*! - If none of the message contain valid nav data, unvalidate the nav and populate a zero message */
    if (opNavIn1.valid == 0 && opNavIn2.valid == 0){
        opNavMsgOut.valid = 0;
        OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
    }
    /*! - Only one of two are valid */
    else if (opNavIn1.valid == 1 && opNavIn2.valid == 0){
        /*! - Only one of two are valid */
        if (configData->faultMode<2){
            opNavMsgOut = opNavIn1;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
        else{
            opNavMsgOut.valid = 0;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
    }
    else if (opNavIn1.valid == 0 && opNavIn2.valid == 1){
        /*! - If secondary measurments are trusted use them as primary */
        if (configData->faultMode==0){
            opNavMsgOut = opNavIn2;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
        /*! - If secondaries are not trusted, do not risk corrupting measurment */
        if (configData->faultMode>0){
            opNavMsgOut.valid = 0;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
    }
    /*! - If they are both valid proceed to the fault detection */
    else if (opNavIn1.valid == 1 && opNavIn2.valid == 1){
        opNavMsgOut.valid = 1;
        /*! -- Dissimilar mode compares the two measurements: if the mean +/- 3-sigma covariances overlap use the nominal nav solution */
        double faultDirection[3];
        double faultNorm;
        
        /*! Get the direction and norm between the the two measurements in camera frame*/
        v3Subtract(opNavIn2.r_BN_C, opNavIn1.r_BN_C, faultDirection);
        faultNorm = v3Norm(faultDirection);
        v3Normalize(faultDirection, faultDirection);
        
        /*! If the difference between vectors is beyond the covariances, detect a fault and use secondary */
        if (faultNorm > configData->sigmaFault*sqrt((vNorm(opNavIn1.covar_C, 9) + vNorm(opNavIn2.covar_C, 9)))){
            opNavMsgOut = opNavIn2;
            opNavMsgOut.faultDetected = 1;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
        /*! If the difference between vectors is low, use primary */
        else if (configData->faultMode>0){
            /*! Bring all the measurements and covariances into their respective frames */
            opNavMsgOut = opNavIn1;
            OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        }
        /*! -- Merge mode combines the two measurements and uncertainties if they are similar */
        else if (configData->faultMode==0){
            double P1invX1[3], P2invX2[3], X_C[3], P_C[3][3], P_B[3][3], P_N[3][3];
            double P1inv[3][3], P2inv[3][3];
            
            /*! The covariance merge is given by P = (P1^{-1} + P2^{-1})^{-1} */
            m33Inverse(RECAST3X3 opNavIn1.covar_C, P1inv);
            m33Inverse(RECAST3X3 opNavIn2.covar_C, P2inv);
            m33Add(P1inv, P2inv, P_C);
            m33Inverse(P_C, P_C);

            /*! The estimage merge is given by x = P (P1^{-1}x1 + P2^{-1}x2) */
            m33MultV3(P1inv, opNavIn1.r_BN_C, P1invX1);
            m33MultV3(P2inv, opNavIn2.r_BN_C, P2invX2);
            v3Add(P1invX1, P2invX2, X_C);
            m33MultV3(P_C, X_C, X_C);
            v3Copy(X_C, opNavMsgOut.r_BN_C);
            
            /*! Bring all the measurements and covariances into their respective frames */
            m33MultV3(dcm_NC, X_C, opNavMsgOut.r_BN_N);
            mCopy(P_C, 3, 3, opNavMsgOut.covar_C);
            m33tMultM33(dcm_CB, P_C, P_B);
            m33MultM33(dcm_NC, P_C, P_N);
            m33MultM33(P_B, dcm_CB, P_B);
            m33MultM33t(P_N, dcm_NC , P_N);
            mCopy(P_N, 3, 3, opNavMsgOut.covar_N);
            mCopy(P_B, 3, 3, opNavMsgOut.covar_B);
            m33tMultV3(dcm_CB, X_C, opNavMsgOut.r_BN_B);
            m33MultV3(dcm_NC, X_C, opNavMsgOut.r_BN_N);
            opNavMsgOut.timeTag = opNavIn1.timeTag;
            opNavMsgOut.planetID = opNavIn1.planetID;
        }
        OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
    }

    
    return;

}
