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
/*
    Simple Thruster Force Evaluation from
 */

#include "fswAlgorithms/effectorInterfaces/thrForceMapping/thrForceMapping.h"
#include "architecture/utilities/macroDefinitions.h"
#include <string.h>
#include <math.h>
#include "architecture/utilities/linearAlgebra.h"


/*! self init method
 @return void
 @param configData The configuration data associated with this module
 @param moduleID The ID associated with the configData
 */
void SelfInit_thrForceMapping(thrForceMappingConfig *configData, int64_t moduleID)
{
    THRArrayCmdForceMsg_C_init(&configData->thrForceCmdOutMsg);
}


/*! This method performs a complete reset of the module.  Local module variables that retain
 time varying states between function calls are reset to their default values.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Reset_thrForceMapping(thrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    double             *pAxis;                  /* pointer to the current control axis */
    uint32_t                 i;
    THRArrayConfigMsgPayload   localThrusterData;   /* local copy of the thruster data message */

    /*! - configure the number of axes that are controlled */
    configData->numControlAxes = 0;
    for (i=0;i<3;i++)
    {
        pAxis = configData->controlAxes_B + 3*configData->numControlAxes;
        if (v3Norm(pAxis) > configData->epsilon) {
            v3Normalize(pAxis,pAxis);
            configData->numControlAxes += 1;
        } else {
            break;
        }
    }
    if (configData->numControlAxes==0) {
        _bskLog(configData->bskLogger, BSK_ERROR,"thrForceMapping() is not setup to control any axes!");
    }
    if (configData->thrForceSign==0) {
        _bskLog(configData->bskLogger, BSK_ERROR,"thrForceMapping() must have thrForceSign set to either +1 or -1");
    }

    // check if the required input messages are included
    if (!THRArrayConfigMsg_C_isLinked(&configData->thrConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrForceMapping.thrConfigInMsg wasn't connected.");
    }
    if (!VehicleConfigMsg_C_isLinked(&configData->vehConfigInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrForceMapping.vehConfigInMsg wasn't connected.");
    }
    if (!CmdTorqueBodyMsg_C_isLinked(&configData->cmdTorqueInMsg)) {
        _bskLog(configData->bskLogger, BSK_ERROR, "Error: thrForceMapping.cmdTorqueInMsg wasn't connected.");
    }

    /*! - read in the support thruster and vehicle configuration messages */
    localThrusterData = THRArrayConfigMsg_C_read(&configData->thrConfigInMsg);
    configData->sc = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);

    /*! - copy the thruster position and thruster force heading information into the module configuration data */
    configData->numThrusters = (uint32_t) localThrusterData.numThrusters;
    for(i=0; i<configData->numThrusters; i=i+1)
    {
        v3Copy(localThrusterData.thrusters[i].rThrust_B, configData->rThruster_B[i]);
        v3Copy(localThrusterData.thrusters[i].tHatThrust_B, configData->gtThruster_B[i]);
        if(localThrusterData.thrusters[i].maxThrust <= 0.0){
            _bskLog(configData->bskLogger, BSK_ERROR, "A configured thruster has a non-sensible saturation limit of <= 0 N!");
        } else {
            configData->thrForcMag[i] = localThrusterData.thrusters[i].maxThrust;
        }
    }
}

/*! The module takes a body frame torque vector and projects it onto available RCS or DV thrusters.
 @return void
 @param configData The configuration data associated with the module
 @param callTime The clock time at which the function was called (nanoseconds)
 @param moduleID The ID associated with the configData
 */
void Update_thrForceMapping(thrForceMappingConfig *configData, uint64_t callTime, int64_t moduleID)
{
    int         counterPosForces;             /* []      counter for number of positive thruster forces */
    double      F[MAX_EFF_CNT];               /* [N]     vector of commanded thruster forces */
    double      Fbar[MAX_EFF_CNT];            /* [N]     vector of intermediate thruster forces */
    double      D[3][MAX_EFF_CNT];            /* [m]     mapping matrix from thruster forces to body torque */
    double      Dbar[3][MAX_EFF_CNT];         /* [m]     reduced mapping matrix*/
    double      C[3][3];                      /* [m]     control mapping matrix*/
    double      Lr_B[3];                      /* [Nm]    commanded ADCS control torque */
    double      Lr_offset[3];
    double      LrLocal[3];                   /* [Nm]    Torque provided by indiviual thruster */
    int         thrusterUsed[MAX_EFF_CNT];    /* []      Array of flags indicating if this thruster is used for the Lr_j */
    double      rThrusterRelCOM_B[MAX_EFF_CNT][3];/* [m]     local copy of the thruster locations relative to COM */
    double      Lr_B_Bar[3];                     /* [Nm]    Control torque that we actually control*/
    double      maxFractUse;                  /* []      ratio of maximum requested thruster force relative to maximum thruster limit */
    double      rCrossGt[3];
    CmdTorqueBodyMsgPayload LrInputMsg;
    THRArrayCmdForceMsgPayload thrusterForceOut;

    /*! - zero all output message copies */
    thrusterForceOut = THRArrayCmdForceMsg_C_zeroMsgPayload();
    
    /*! - clear arrays of the thruster mapping algorithm */
    vSetZero(F, MAX_EFF_CNT);
    mSetZero(D, 3, MAX_EFF_CNT);
    mSetZero(Dbar, 3, MAX_EFF_CNT);
    mSetZero(C, 3, 3);

    /*! - Read the input messages */
    LrInputMsg = CmdTorqueBodyMsg_C_read(&configData->cmdTorqueInMsg);
    configData->sc = VehicleConfigMsg_C_read(&configData->vehConfigInMsg);

    /*! - copy the request 3D attitude control torque vector */
    v3Copy(LrInputMsg.torqueRequestBody, Lr_B);

    /*! - compute thruster locations relative to COM */
    for (uint32_t i=0;i<configData->numThrusters;i++) {
        v3Subtract(configData->rThruster_B[i], configData->sc.CoM_B, rThrusterRelCOM_B[i]); /* Part 1 of Eq. 4 */
    }
   
    /*! - compute general thruster force mapping matrix */
    v3SetZero(Lr_offset);

    for(uint32_t i=0; i<configData->numThrusters; i=i+1)
    {
        v3Cross(rThrusterRelCOM_B[i], configData->gtThruster_B[i], rCrossGt); /* Eq. 6 */
        for(uint32_t j=0; j<3; j++)
        {
            D[j][i] = rCrossGt[j];
        }
        if(configData->thrForceSign < 0)  /* Handles the case where there is translational motion imparted during off-pulsing*/
        {
            v3Scale(configData->thrForcMag[i], rCrossGt, LrLocal); /* Computing local torques from each thruster -- Individual terms in Eq. 7*/
            v3Subtract(Lr_offset, LrLocal, Lr_offset); /* Summing of individual torques -- Eq. 5 & Eq. 7 */
        }
    }
    
    v3Add(Lr_offset, Lr_B, Lr_B);
    
    /*! - copy the control axes into [C] */
    for (uint32_t i=0;i<configData->numControlAxes;i++) {
        v3Copy(&configData->controlAxes_B[3*i], C[i]);
    }
    
    /*! - map the control torque onto the control axes*/
    m33MultV3(RECAST3X3 C, Lr_B, Lr_B_Bar); /* Note: Lr_B_Bar is projected only onto the available control axes. i.e. if using DV thrusters with only 1 control axis, Lr_B_Bar = [#, 0, 0] */

    /*! - 1st iteration of finding a set of force vectors to implement the control torque */
    findMinimumNormForce(configData, D, Lr_B_Bar, configData->numThrusters, F);
    
    /*! - Remove forces components that are contributing to the RCS Null space (this is due to the geometry of the thrusters) */
    if (configData->thrForceSign>0)
    {
        substractMin(F, configData->numThrusters);
    }
    
    if (configData->thrForceSign<0 || configData->use2ndLoop)
    {
        counterPosForces = 0;
        memset(thrusterUsed,0x0,MAX_EFF_CNT*sizeof(int));
        for (uint32_t i=0;i<configData->numThrusters;i++) {
            if (F[i]*configData->thrForceSign > 0) {
                thrusterUsed[i] = 1; /* Eq. 11 */
                for(uint32_t j=0; j<3; j++)
                {
                    Dbar[j][counterPosForces] = D[j][i]; /* Eq. 12 */
                }
                counterPosForces += 1;
            }
        }

        findMinimumNormForce(configData, Dbar, Lr_B_Bar, counterPosForces, Fbar);
        if (configData->thrForceSign > 0)
        {
            substractMin(Fbar, counterPosForces);
        }
        uint32_t c = 0;
        for (uint32_t i=0;i<configData->numThrusters;i++) {
            if (thrusterUsed[i]) {
                F[i] = Fbar[c];
                c += 1;
            } else {
                F[i] = 0.0;
            }
        }
    }
    
    configData->outTorqAngErr = computeTorqueAngErr(D, Lr_B_Bar, configData->numThrusters, configData->epsilon, F,
        configData->thrForcMag); /* Eq. 16*/
    maxFractUse = 0.0;
    /*  check if the angle between the request and actual torque exceeds a limit.  If so, then uniformly scale
        all thruster forces values to not exceed saturation.
        If the angle threshold is negative, then this scaling is bypassed.*/
    if(configData->outTorqAngErr > configData->angErrThresh)
    {
        for(uint32_t i=0; i<configData->numThrusters; i++)
        {
            if(configData->thrForcMag[i] > 0.0 && fabs(F[i])/configData->thrForcMag[i] > maxFractUse) /* confirming that maxThrust > 0 */
            {
                maxFractUse = fabs(F[i])/configData->thrForcMag[i];
            }
        }
        /* only scale the requested thruster force if one or more thrusters are saturated */
        if(maxFractUse > 1.0)
        {
            vScale(1.0/maxFractUse, F, configData->numThrusters, F);
            configData->outTorqAngErr = computeTorqueAngErr(D, Lr_B_Bar, configData->numThrusters, configData->epsilon, F,
                                                            configData->thrForcMag);
        }
    }

    /* store the output message */
    mCopy(F, configData->numThrusters, 1, thrusterForceOut.thrForce);
    THRArrayCmdForceMsg_C_write(&thrusterForceOut, &configData->thrForceCmdOutMsg, moduleID, callTime);

    return;
}

/*!
 Take a stack of force values find the smallest value, and subtract if from all force values.  Here the smallest values
 will become zero, while other forces increase.  This assumes that the thrusters are aligned such that if all
 thrusters are firing, then no torque or force is applied.  This ensures only positive force values are computed.
 */
void substractMin(double *F, uint32_t size)
{
    uint32_t    i;
    double minValue = 0.0;
    for (i=0; i < size;i++){
        if(F[i] < minValue){
            minValue = F[i];
        }
    }
    for (i=0; i < size;i++){
        F[i] = F[i] - minValue;
    }
    return;
}


/*!
 Use a least square inverse to determine the smallest set of thruster forces that yield the desired torque vector.  Note
 that this routine does not constrain yet the forces to be either positive or negative
 */
void findMinimumNormForce(thrForceMappingConfig *configData,
                          double D[3][MAX_EFF_CNT], double Lr_B_Bar[3], uint32_t numForces, double F[MAX_EFF_CNT])
{
    
    uint32_t         i,j,k;                          /* []     counters */
    double      C[3][3];                        /* [m^2]  (C) matrix */
    double      CD[3][MAX_EFF_CNT];             /* [m^2]  [C].[D] matrix -- Thrusters in body frame mapped on control axes */
    double      CDCDT[3][3];                    /* [m^2]  [CD].[CD]^T matrix */
    double      CDCDTInv[3][3];                 /* [m^2]  ([CD].[CD]^T)^-1 matrix */
    double      CDCDTInvLr[3];

    vSetZero(F, MAX_EFF_CNT);   /* zero the output force vector */
    m33SetZero(C);              /* zero the control basis */
    
    /*! - copy the control axes into [C] */
    for (i=0;i<configData->numControlAxes;i++) {
        v3Copy(&configData->controlAxes_B[3*i], C[i]);
    }

    /* find [D].[D]^T */
    mMultM(C, 3, 3, D, 3, MAX_EFF_CNT, CD);
    m33SetIdentity(CDCDT);
    for(i=0; i<configData->numControlAxes; i++) {
        for(j=0; j<configData->numControlAxes; j++) {
            CDCDT[i][j] = 0.0;
            for (k=0;k<numForces;k++) {
                CDCDT[i][j] += CD[i][k] * CD[j][k]; /* Part of Eq. 9 */
            }
        }
    }
    
    if (m33Determinant(CDCDT) > configData->epsilon){
        m33Inverse(CDCDT, CDCDTInv);
    } else {
        m33SetZero(CDCDTInv);
    }
        
    m33MultV3(CDCDTInv, Lr_B_Bar, CDCDTInvLr);/* If fewer than 3 control axes, then the 1's along the diagonal of DDTInv will not conflict with the mapping, as Lr_B_Bar contains the nessessary 0s to inhibit projection */
    mtMultV(CD, 3, MAX_EFF_CNT, CDCDTInvLr, F); /* Eq. 15 */

    return;
}

/*!
 Determine the angle between the desired torque vector and the actual torque vector.
 */
double computeTorqueAngErr(double D[3][MAX_EFF_CNT], double BLr_B[3], uint32_t numForces, double epsilon,
                           double F[MAX_EFF_CNT], double FMag[MAX_EFF_CNT])
{
    double returnAngle = 0.0;       /* [rad]  angle between requested and actual torque vector */
    /*! - make sure a control torque is requested, otherwise just return a zero angle error */
    if (v3Norm(BLr_B) > epsilon) {
        
        double tauActual_B[3];          /* [Nm]   control torque with current thruster solution */
        double BLr_hat_B[3];            /* []     normalized BLr_B vector */
        double LrEffector_B[3];         /* [Nm]   torque of an individual thruster effector */
        double thrusterForce;           /* [N]    saturation constrained thruster force */
        
        double DT[MAX_EFF_CNT][3];
        mTranspose(D, 3, MAX_EFF_CNT, DT);
        v3Normalize(BLr_B, BLr_hat_B);
        v3SetZero(tauActual_B);

        /*! - loop over all thrusters and compute the actual torque to be applied */
        for(uint32_t i=0; i<numForces; i++)
        {
            thrusterForce = fabs(F[i]) < FMag[i] ? F[i] : FMag[i]*fabs(F[i])/F[i]; /* This could produce inf's as F[i] approaches 0 if FMag[i] is 0, as such we check if FMag[i] is equal to zero in reset() */
            v3Scale(thrusterForce, DT[i], LrEffector_B);
            v3Add(tauActual_B, LrEffector_B, tauActual_B);
        }

        /*! - evaluate the angle between the requested and thruster implemented torque vector */
        v3Normalize(tauActual_B, tauActual_B);
        if(v3Dot(BLr_hat_B, tauActual_B) < 1.0)
        {
            returnAngle = safeAcos(v3Dot(BLr_hat_B, tauActual_B)); /* Eq 16 */
        }
    }
    return(returnAngle);
    
}
