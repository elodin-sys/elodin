/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#include "hillToAttRef.h"
#include "string.h"
#include "math.h"
#include "architecture/utilities/linearAlgebra.h"
#include "architecture/utilities/rigidBodyKinematics.h"

void SelfInit_hillToAttRef(HillToAttRefConfig *configData, int64_t moduleID){
    AttRefMsg_C_init(&configData->attRefOutMsg);
}

/*! This method performs a complete reset of the module.  Local module variables that retain time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Reset_hillToAttRef(HillToAttRefConfig *configData,  uint64_t callTime, int64_t moduleID)
{
    if (!HillRelStateMsg_C_isLinked(&configData->hillStateInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: hillToAttRef.hillStateInMsg wasn't connected.");
    }
    
    if (AttRefMsg_C_isLinked(&configData->attRefInMsg) && NavAttMsg_C_isLinked(&configData->attNavInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: hillToAttRef can't have both attRefInMsg and attNavInMsg connected.");
    }

    if (!AttRefMsg_C_isLinked(&configData->attRefInMsg) && !NavAttMsg_C_isLinked(&configData->attNavInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: hillToAttRef must have one of attRefInMsg and attNavInMsg connected.");
    }

    return;
}

AttRefMsgPayload RelativeToInertialMRP(HillToAttRefConfig *configData, double relativeAtt[3], double sigma_XN[3]){
    AttRefMsgPayload attRefOut;
    //  Check to see if the relative attitude components exceed specified bounds (by default these are non-physical and should never be reached)
    for(int ind=0; ind<3; ++ind){
        relativeAtt[ind] = fmax(relativeAtt[ind], configData->relMRPMin);
        relativeAtt[ind] = fmin(relativeAtt[ind], configData->relMRPMax);
    }

    //  Combine the relative attitude with the chief inertial attitude to get the reference attitude
    addMRP(sigma_XN, relativeAtt, attRefOut.sigma_RN);
    for(int ind=0; ind<3; ++ind){
        attRefOut.omega_RN_N[ind] = 0;
        attRefOut.domega_RN_N[ind] = 0;
    }
    return(attRefOut);
}

/*! This module reads an OpNav image and extracts circle information from its content using OpenCV's HoughCircle Transform. It performs a greyscale, a bur, and a threshold on the image to facilitate circle-finding. 
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The Basilisk module identifier
 */
void Update_hillToAttRef(HillToAttRefConfig *configData, uint64_t callTime, int64_t moduleID) {

    HillRelStateMsgPayload hillStateInPayload;
    NavAttMsgPayload attStateInPayload;
    AttRefMsgPayload attRefInPayload;
    AttRefMsgPayload attRefOutPayload;
    
    double baseSigma[3];
    double relativeAtt[3];
    double hillState[6];

    // Do message reads
    hillStateInPayload = HillRelStateMsg_C_read(&configData->hillStateInMsg);
    if(AttRefMsg_C_isLinked(&configData->attRefInMsg)){
        attRefInPayload = AttRefMsg_C_read(&configData->attRefInMsg);
        v3Copy(attRefInPayload.sigma_RN, baseSigma);
    }
    else if(NavAttMsg_C_isLinked(&configData->attNavInMsg)){
        attStateInPayload = NavAttMsg_C_read(&configData->attNavInMsg);
        v3Copy(attStateInPayload.sigma_BN, baseSigma);
    }

    //  Create a state vector based on the current Hill positions
    for(int ind=0; ind<3; ind++){
        hillState[ind] = hillStateInPayload.r_DC_H[ind];
        hillState[ind+3] = hillStateInPayload.v_DC_H[ind];
    }

    // std::cout<<"Current relative state: "<<hillState[0]<<" "<<hillState[1]<<" "<<hillState[2]<<" "<<hillState[3]<<" "<<hillState[4]<<" "<<hillState[5]<<std::endl;
    // std::cout<<"Printing current gain matrix:"<<std::endl;
    // std::cout<<gainMat[0][0]<<" "<<gainMat[0][1]<<" "<<gainMat[0][2]<<" "<<gainMat[0][3]<<" "<<gainMat[0][4]<<" "<<gainMat[0][5]<<std::endl;
    // std::cout<<gainMat[1][0]<<" "<<gainMat[1][1]<<" "<<gainMat[1][2]<<" "<<gainMat[1][3]<<" "<<gainMat[1][4]<<" "<<gainMat[1][5]<<std::endl;
    // std::cout<<gainMat[2][0]<<" "<<gainMat[2][1]<<" "<<gainMat[2][2]<<" "<<gainMat[2][3]<<" "<<gainMat[2][4]<<" "<<gainMat[2][5]<<std::endl;

    //  Apply the gainMat to the relative state to produce a chief-relative attitude
    mMultV(&configData->gainMatrix, 3, 6,
                   hillState,
                   relativeAtt);
                   
    // std::cout<<"Relative att components: "<<relativeAtt[0]<<" "<<relativeAtt[1]<<" "<<relativeAtt[2]<<std::endl;
    //  Convert that to an inertial attitude and write the attRef msg
    attRefOutPayload = RelativeToInertialMRP(configData, relativeAtt, baseSigma);
    AttRefMsg_C_write(&attRefOutPayload, &configData->attRefOutMsg, moduleID, callTime);

    // this->matrixIndex += 1;
}

