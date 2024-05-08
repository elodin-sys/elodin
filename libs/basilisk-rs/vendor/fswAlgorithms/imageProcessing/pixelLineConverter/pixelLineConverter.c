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
#include "pixelLineConverter.h"



/*! This method transforms pixel, line, and diameter data into heading data for orbit determination or heading determination.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param moduleID The module identification integer
 */
void SelfInit_pixelLineConverter(PixelLineConvertData *configData, int64_t moduleID)
{
    OpNavMsg_C_init(&configData->opNavOutMsg);
}


/*! This resets the module to original states.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Reset_pixelLineConverter(PixelLineConvertData *configData, uint64_t callTime, int64_t moduleID)
{
    // check that the required message has not been connected
    if (!CameraConfigMsg_C_isLinked(&configData->cameraConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineConverter.cameraConfigInMsg wasn't connected.");
    }
    if (!OpNavCirclesMsg_C_isLinked(&configData->circlesInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineConverter.circlesInMsg wasn't connected.");
    }
    if (!NavAttMsg_C_isLinked(&configData->attInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: pixelLineConverter.attInMsg wasn't connected.");
    }

}

/*! This method reads in the camera and circle messages and extracts navigation data from them. It outputs the heading (norm and direction) to the celestial body identified in the inertial frame. It provides the heading to the most robust circle identified by the image processing algorithm.
 @return void
 @param configData The configuration data associated with the ephemeris model
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The module identification integer
 */
void Update_pixelLineConverter(PixelLineConvertData *configData, uint64_t callTime, int64_t moduleID)
{
    double dcm_NC[3][3], dcm_CB[3][3], dcm_BN[3][3];
    double reCentered[2];
    CameraConfigMsgPayload cameraSpecs;
    OpNavCirclesMsgPayload circlesIn;
    OpNavMsgPayload opNavMsgOut;
    NavAttMsgPayload attInfo;

    opNavMsgOut = OpNavMsg_C_zeroMsgPayload();

    /*! - read input messages */
    cameraSpecs = CameraConfigMsg_C_read(&configData->cameraConfigInMsg);
    circlesIn = OpNavCirclesMsg_C_read(&configData->circlesInMsg);
    attInfo = NavAttMsg_C_read(&configData->attInMsg);
    
    if (circlesIn.valid == 0){
        opNavMsgOut.valid = 0;
        OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);
        return;
    }
    reCentered[0] = circlesIn.circlesCenters[0] - cameraSpecs.resolution[0]/2 + 0.5;
    reCentered[1] = circlesIn.circlesCenters[1] - cameraSpecs.resolution[1]/2 + 0.5;
    configData->planetTarget = (int32_t) circlesIn.planetIds[0];
    MRP2C(cameraSpecs.sigma_CB, dcm_CB);
    MRP2C(attInfo.sigma_BN, dcm_BN);
    m33MultM33(dcm_CB, dcm_BN, dcm_NC);
    m33Transpose(dcm_NC, dcm_NC);

    /*! - Find pixel size using camera specs */
    double X, Y;
    double pX, pY;
    /* compute sensorSize/focalLength = 2*tan(FOV/2) */
    pX = 2.*tan(cameraSpecs.fieldOfView*cameraSpecs.resolution[0]/cameraSpecs.resolution[1]/2.0);
    pY = 2.*tan(cameraSpecs.fieldOfView/2.0);
    X = pX/cameraSpecs.resolution[0];
    Y = pY/cameraSpecs.resolution[1];

    /*! - Get the heading */
    double rtilde_C[2];
    double rHat_BN_C[3], rHat_BN_N[3], rHat_BN_B[3];
    double rNorm = 1;
    double planetRad, denom;
    double covar_map_C[3*3], covar_In_C[3*3], covar_In_B[3*3];
    double covar_In_N[3*3];
    double x_map, y_map, rho_map;
    rtilde_C[0] = reCentered[0]*X;
    rtilde_C[1] = reCentered[1]*Y;
    v3Set(rtilde_C[0], rtilde_C[1], 1.0, rHat_BN_C);
    v3Scale(-1, rHat_BN_C, rHat_BN_C);
    v3Normalize(rHat_BN_C, rHat_BN_C);
    
    m33MultV3(dcm_NC, rHat_BN_C, rHat_BN_N);
    m33tMultV3(dcm_CB, rHat_BN_C, rHat_BN_B);

    if(configData->planetTarget > 0){
        if(configData->planetTarget ==1){
            planetRad = REQ_EARTH;//in km
            opNavMsgOut.planetID = configData->planetTarget;
        }
        if(configData->planetTarget ==2){
            planetRad = REQ_MARS;//in km
            opNavMsgOut.planetID = configData->planetTarget;
        }
        if(configData->planetTarget ==3){
            planetRad = REQ_JUPITER;//in km
            opNavMsgOut.planetID = configData->planetTarget;
        }
        
        denom = sin(atan(X*circlesIn.circlesRadii[0]));
        rNorm = planetRad/denom; //in km
        
        /*! - Compute the uncertainty */
        x_map = planetRad/denom*(X);
        y_map = planetRad/denom*(Y);
        rho_map = planetRad*(X/(sqrt(1 + pow(circlesIn.circlesRadii[0]*X,2)))-1.0/X*sqrt(1 + pow(circlesIn.circlesRadii[0]*X,2))/pow(circlesIn.circlesRadii[0], 2));
        mSetIdentity(covar_map_C, 3, 3);
        covar_map_C[0] = x_map;
        covar_map_C[4] = y_map;
        covar_map_C[8] = rho_map;
        mCopy(circlesIn.uncertainty, 3, 3, covar_In_C);
        mMultM(covar_map_C, 3, 3, covar_In_C, 3, 3, covar_In_C);
        mMultMt(covar_In_C, 3, 3, covar_map_C, 3, 3, covar_In_C);
        /*! - Changer the mapped covariance to inertial frame */
        mMultM(dcm_NC, 3, 3, covar_In_C, 3, 3, covar_In_N);
        mMultMt(covar_In_N, 3, 3, dcm_NC, 3, 3, covar_In_N);
        /*! - Changer the mapped covariance to body frame */
        mtMultM(dcm_CB, 3, 3, covar_In_C, 3, 3, covar_In_B);
        mMultM(covar_In_B, 3, 3, dcm_CB, 3, 3, covar_In_B);
    }
    
    /*! - write output message */
    v3Scale(rNorm*1E3, rHat_BN_N, opNavMsgOut.r_BN_N); //in m
    v3Scale(rNorm*1E3, rHat_BN_C, opNavMsgOut.r_BN_C); //in m
    v3Scale(rNorm*1E3, rHat_BN_B, opNavMsgOut.r_BN_B); //in m
    mCopy(covar_In_N, 3, 3, opNavMsgOut.covar_N);
    vScale(1E6, opNavMsgOut.covar_N, 3*3, opNavMsgOut.covar_N);//in m
    mCopy(covar_In_C, 3, 3, opNavMsgOut.covar_C);
    vScale(1E6, opNavMsgOut.covar_C, 3*3, opNavMsgOut.covar_C);//in m
    mCopy(covar_In_B, 3, 3, opNavMsgOut.covar_B);
    vScale(1E6, opNavMsgOut.covar_B, 3*3, opNavMsgOut.covar_B);//in m
    opNavMsgOut.timeTag = (double) circlesIn.timeTag;
    opNavMsgOut.valid =1;

    OpNavMsg_C_write(&opNavMsgOut, &configData->opNavOutMsg, moduleID, callTime);

    return;
}
