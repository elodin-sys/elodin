''' '''
'''
 ISC License

 Copyright (c) 2016-2017, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

'''

#   This test validates the EKF module by running several
#   scenarios on both individual functions and the full module.
#   Author: Thibaud Teil

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import sunlineSEKF
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros, RigidBodyKinematics
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions

import SunLineSEKF_test_utilities as FilterPlots


def addTimeColumn(time, data):
    return np.transpose(np.vstack([[time], np.transpose(data)]))


def setupFilterData(filterObject):

    filterObject.sensorUseThresh = 0.
    filterObject.state = [0.1, 0.9, 0.1, 0.0, 0.0]
    filterObject.x = [1.0, 0.0, 1.0, 0.0, 0.1]
    filterObject.covar = [0.4, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.4, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.4, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.004, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.004]

    filterObject.qProcVal = 0.1**2
    filterObject.qObsVal = 0.001
    filterObject.eKFSwitch = (4./3)**2 #If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF


def test_all_functions_sekf(show_plots):
    """Module Unit Test"""
    [testResults, testMessage] = sunline_individual_test()
    assert testResults < 1, testMessage
    [testResults, testMessage] = StatePropStatic()
    assert testResults < 1, testMessage
    [testResults, testMessage] = StatePropVariable(show_plots)
    assert testResults < 1, testMessage

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(True)

# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("SimHalfLength, AddMeasNoise , testVector1 , testVector2, stateGuess", [
    (200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0]),
    (2000, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0]),
    (200, False ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0]),
    (200, False ,[0., 1., 0.] ,[1., 0., 0.], [0.3, 0.0, 0.6, 0.0, 0.0]),
    (200, True ,[0.5, 0.5, 0.] ,[0., 1., 0.], [0.7, 0.7, 0.0, 0.0, 0.0])
])


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def test_all_sunline_sekf(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess):
    [testResults, testMessage] = StateUpdateSunLine(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess)
    assert testResults < 1, testMessage


def sunline_individual_test():
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    numStates = 5
    numObs = 3

    ###################################################################################
    ## Testing dynamics matrix computation
    ###################################################################################
    inputStates = [2,1,0.75,0.1,0.4]
    inputOmega_SB_S = [0.,0.1, 0.4]
    bVec = [1.,0.,0.]
    dt =0.5

    dcm_BS = [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]

    # Fill in the variables for the test
    dcm = sunlineSEKF.new_doubleArray(3 * 3)
    for j in range(9):
        sunlineSEKF.doubleArray_setitem(dcm, j, dcm_BS[j])
    sunlineSEKF.sunlineSEKFComputeDCM_BS(inputStates[:3], bVec, dcm)

    dcmOut = []
    for j in range(9):
        dcmOut.append(sunlineSEKF.doubleArray_getitem(dcm, j))

    DCM_BS = np.array(dcmOut).reshape([3,3])

    omega_SB_B = np.dot(DCM_BS, np.array(inputOmega_SB_S))
    dtilde = RigidBodyKinematics.v3Tilde(np.array(inputStates)[:3])
    dBS = np.dot(dtilde, DCM_BS)

    expDynMat = np.zeros([numStates,numStates])
    expDynMat[0:3, 0:3] =  np.array(RigidBodyKinematics.v3Tilde(omega_SB_B))
    expDynMat[0:3, 3:numStates] = -dBS[:, 1:]

    dynMat = sunlineSEKF.new_doubleArray(numStates*numStates)
    for i in range(numStates*numStates):
        sunlineSEKF.doubleArray_setitem(dynMat, i, 0.0)
    sunlineSEKF.sunlineDynMatrix(inputStates, bVec, dt, dynMat)

    DynOut = []
    for i in range(numStates*numStates):
        DynOut.append(sunlineSEKF.doubleArray_getitem(dynMat, i))

    DynOut = np.array(DynOut).reshape(numStates, numStates)
    errorNorm = np.linalg.norm(expDynMat - DynOut)
    if(errorNorm > 1.0E-10):
        print(errorNorm, "Dyn Matrix")
        testFailCount += 1
        testMessages.append("Dynamics Matrix generation Failure Dyn " + "\n")

    ###################################################################################
    ## STM and State Test
    ###################################################################################

    inputStates = [2,1,0.75,0.1,0.4]
    inputOmega = [0.,0.1, 0.4]
    bVec_test = [1,0,0]
    dt = 0.5
    stateTransition = sunlineSEKF.new_doubleArray(numStates*numStates)
    states = sunlineSEKF.new_doubleArray(numStates)
    bVec = sunlineSEKF.new_doubleArray(3)
    for k in range(3):
        sunlineSEKF.doubleArray_setitem(bVec, k, bVec_test[k])
    for i in range(numStates):
        sunlineSEKF.doubleArray_setitem(states, i, inputStates[i])
        for j in range(numStates):
            if i==j:
                sunlineSEKF.doubleArray_setitem(stateTransition, numStates*i+j, 1.0)
            else:
                sunlineSEKF.doubleArray_setitem(stateTransition, numStates*i+j, 0.0)

    sunlineSEKF.sunlineStateSTMProp(expDynMat.flatten().tolist(), bVec_test, dt, states, stateTransition)

    PropStateOut = []
    PropSTMOut = []
    for i in range(numStates):
        PropStateOut.append(sunlineSEKF.doubleArray_getitem(states, i))
    for i in range(numStates*numStates):
        PropSTMOut.append(sunlineSEKF.doubleArray_getitem(stateTransition, i))

    dcm_BS = [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]

    # Fill in the variables for the test
    dcm = sunlineSEKF.new_doubleArray(3 * 3)

    for j in range(9):
        sunlineSEKF.doubleArray_setitem(dcm, j, dcm_BS[j])

    sunlineSEKF.sunlineSEKFComputeDCM_BS(inputStates[:3], bVec_test, dcm)

    dcmOut = []
    for j in range(9):
        dcmOut.append(sunlineSEKF.doubleArray_getitem(dcm, j))

    DCM_BS = np.array(dcmOut).reshape([3,3])
    STMout = np.array(PropSTMOut).reshape([numStates,numStates])
    StatesOut = np.array(PropStateOut)

    expectedSTM = dt*np.dot(expDynMat, np.eye(numStates)) + np.eye(numStates)
    expectedStates = np.zeros(numStates)
    ## Equations when removing the unobservable states from d_dot
    expectedStates[3:numStates] = np.array(inputOmega)[1:3]
    expectedStates[0:3] = np.array(inputStates)[0:3]+dt*np.cross(np.dot(DCM_BS,np.array(inputOmega)), np.array(inputStates)[0:3])
    errorNormSTM = np.linalg.norm(expectedSTM - STMout)
    errorNormStates = np.linalg.norm(expectedStates - StatesOut)

    if(errorNormSTM > 1.0E-10):
        testFailCount += 1
        testMessages.append("STM Propagation Failure Dyn "  + "\n")

    if(errorNormStates > 1.0E-10):
        testFailCount += 1
        testMessages.append("State Propagation Failure Dyn " + "\n")

    ###################################################################################
    ## Test the H and yMeas matrix generation as well as the observation count
    ###################################################################################

    numCSS = 4
    cssCos = [np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)), np.cos(np.deg2rad(90.))]
    sensorTresh = np.cos(np.deg2rad(50.))
    cssNormals = [1.,0.,0.,0.,1.,0., 0.,0.,1., 1./np.sqrt(2), 1./np.sqrt(2),0.]
    dcmArray_BS = RigidBodyKinematics.MRP2C([0.1,-0.15,0.2])
    dcm_BS = (dcmArray_BS.flatten()).tolist()

    measMat = sunlineSEKF.new_doubleArray(8*numStates)
    obs = sunlineSEKF.new_doubleArray(8)
    yMeas = sunlineSEKF.new_doubleArray(8)
    numObs = sunlineSEKF.new_intArray(1)

    for i in range(8*numStates):
        sunlineSEKF.doubleArray_setitem(measMat, i, 0.)
    for i in range(8):
        sunlineSEKF.doubleArray_setitem(obs, i, 0.0)
        sunlineSEKF.doubleArray_setitem(yMeas, i, 0.0)

    sunlineSEKF.sunlineHMatrixYMeas(inputStates, numCSS, cssCos, sensorTresh, cssNormals, obs, yMeas, numObs, measMat)

    obsOut = []
    yMeasOut = []
    numObsOut = []
    HOut = []
    for i in range(8*numStates):
        HOut.append(sunlineSEKF.doubleArray_getitem(measMat, i))
    for i in range(8):
        yMeasOut.append(sunlineSEKF.doubleArray_getitem(yMeas, i))
        obsOut.append(sunlineSEKF.doubleArray_getitem(obs, i))
    numObsOut.append(sunlineSEKF.intArray_getitem(numObs, 0))

    #Fill in expected values for test
    expectedH = np.zeros([8,numStates])
    expectedY = np.zeros(8)
    for j in range(3):
        expectedH[j,0:3] = np.eye(3)[j,:]
        expectedY[j] =np.array(cssCos[j]) - np.dot( np.array(inputStates)[0:3], np.array(cssNormals)[j*3:(j+1)*3])
    expectedObs = np.array([np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)),0.,0.,0.,0.,0.])
    expectedNumObs = 3

    HOut = np.array(HOut).reshape([8, numStates])
    errorNorm = np.zeros(4)
    errorNorm[0] = np.linalg.norm(HOut - expectedH)
    errorNorm[1] = np.linalg.norm(yMeasOut - expectedY)
    errorNorm[2] = np.linalg.norm(obsOut - expectedObs)
    errorNorm[3] = np.linalg.norm(numObsOut[0] - expectedNumObs)

    for i in range(4):
        if(errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("H and yMeas update failure \n")

    ###################################################################################
    ## Test the Kalman Gain
    ###################################################################################

    numObs = 3
    h = [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.,
             0., 0., 1., 0., 0.,
             1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.]
    noise= 0.01

    Kalman = sunlineSEKF.new_doubleArray(numStates * 8)

    for i in range(8 * numStates):
        sunlineSEKF.doubleArray_setitem(Kalman, i, 0.)

    sunlineSEKF.sunlineKalmanGain(covar, h, noise, numObs, Kalman)

    KalmanOut = []
    for i in range(8 * numStates):
        KalmanOut.append(sunlineSEKF.doubleArray_getitem(Kalman, i))

    # Fill in expected values for test
    Hmat = np.array(h).reshape([8,numStates])
    Pk = np.array(covar).reshape([numStates,numStates])
    R = noise*np.eye(numObs)
    expectedK = np.dot(np.dot(Pk, Hmat[0:numObs,:].T), np.linalg.inv(np.dot(np.dot(Hmat[0:numObs,:], Pk), Hmat[0:numObs,:].T) + R[0:numObs,0:numObs]))

    KalmanOut = np.array(KalmanOut)[0:numStates*numObs].reshape([numStates, numObs])
    errorNorm = np.linalg.norm(KalmanOut[:,0:numObs] - expectedK)

    if (errorNorm > 1.0E-10):
        print(errorNorm, "Kalman Gain Error")
        testFailCount += 1
        testMessages.append("Kalman Gain update failure \n")

    ###################################################################################
    ## Test the EKF update
    ###################################################################################

    KGain = [1., 2., 3., 0., 1., 1., 0., 1., 0., 1., 3., 0., 1., 0., 2.]
    for i in range(numStates*8-numStates*numObs):
        KGain.append(0.)
    inputStates = [2,1,0.75,0.1,0.4]
    xbar = [0.1, 0.2, 0.01, 0.005, 0.009]
    numObs = 3
    h = [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.,
             0., 0., 1., 0., 0.,
             1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.]
    noise = 0.01
    inputY = np.zeros(3)
    for j in range(3):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3], np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = sunlineSEKF.new_doubleArray(numStates)
    covarMat = sunlineSEKF.new_doubleArray(numStates*numStates)
    inputs = sunlineSEKF.new_doubleArray(numStates)


    for i in range(numStates):
        sunlineSEKF.doubleArray_setitem(stateError, i, 0.)
        sunlineSEKF.doubleArray_setitem(inputs, i, inputStates[i])
        for j in range(numStates):
            sunlineSEKF.doubleArray_setitem(covarMat,i+j,0.)

    sunlineSEKF.sunlineSEKFUpdate(KGain, covar, noise, numObs, inputY, h, inputs, stateError, covarMat)

    stateOut = []
    covarOut = []
    errorOut = []
    for i in range(numStates):
        stateOut.append(sunlineSEKF.doubleArray_getitem(inputs, i))
        errorOut.append(sunlineSEKF.doubleArray_getitem(stateError, i))
    for j in range(numStates*numStates):
        covarOut.append(sunlineSEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:numStates*3].reshape([numStates,3])
    expectedStates = np.array(inputStates) + np.dot(KK, np.array(inputY))
    H = np.array(h).reshape([8,numStates])[0:3,:]
    Pk = np.array(covar).reshape([numStates, numStates])
    R = noise * np.eye(3)
    expectedP = np.dot(np.dot(np.eye(numStates) - np.dot(KK, H), Pk), np.transpose(np.eye(numStates) - np.dot(KK, H))) + np.dot(KK, np.dot(R,KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(stateOut) - expectedStates)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([numStates,numStates]))

    for i in range(2):
        if(errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("EKF update failure \n")

    ###################################################################################
    ## Test the CKF update
    ###################################################################################

    KGain = [1., 2., 3., 0., 1., 1., 0., 1., 0., 1., 3., 0., 1., 0., 2.]
    for i in range(numStates * 8 - numStates * 3):
        KGain.append(0.)
    inputStates = [2,1,0.75,0.1,0.4]
    xbar = [0.1, 0.2, 0.01, 0.005, 0.009]
    h = [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.,
             0., 0., 1., 0., 0.,
             1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.]
    noise =0.01
    inputY = np.zeros(numObs)
    for j in range(numObs):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3],
                                                 np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = sunlineSEKF.new_doubleArray(numStates)
    covarMat = sunlineSEKF.new_doubleArray(numStates * numStates)

    for i in range(numStates):
        sunlineSEKF.doubleArray_setitem(stateError, i, xbar[i])
        for j in range(numStates):
            sunlineSEKF.doubleArray_setitem(covarMat, i + j, 0.)

    sunlineSEKF.sunlineCKFUpdate(xbar, KGain, covar, noise, numObs, inputY, h, stateError, covarMat)

    covarOut = []
    errorOut = []
    for i in range(numStates):
        errorOut.append(sunlineSEKF.doubleArray_getitem(stateError, i))
    for j in range(numStates*numStates):
        covarOut.append(sunlineSEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:numStates * numObs].reshape([numStates, numObs])
    H = np.array(h).reshape([8, numStates])[0:3, :]
    expectedStateError = np.array(xbar) + np.dot(KK, (np.array(inputY) - np.dot(H, np.array(xbar))))
    Pk = np.array(covar).reshape([numStates, numStates])
    expectedP = np.dot(np.dot(np.eye(numStates) - np.dot(KK, H), Pk), np.transpose(np.eye(numStates) - np.dot(KK, H))) + np.dot(KK,
                                                                                                                np.dot(
                                                                                                                    R,
                                                                                                                    KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(errorOut) - expectedStateError)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([numStates, numStates]))

    for i in range(2):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("CKF update failure \n")

    ###################################################################################
    ## Test the sunlineSEKFComputeDCM_BS method
    ###################################################################################

    inputStates = [2, 1, 0.75, 0.1, 0.4]
    sunheading = inputStates[:3]
    bvec1 = [0., 1., 0.]
    b1 = np.array(bvec1)

    dcm_BS = [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]

    # Fill in expected values for test

    DCM_exp = np.zeros([3,3])
    W_exp = np.eye(numStates)

    DCM_exp[:, 0] = np.array(inputStates[0:3]) / (np.linalg.norm(np.array(inputStates[0:3])))
    DCM_exp[:, 1] = np.cross(DCM_exp[:, 0], b1) / np.linalg.norm(np.array(np.cross(DCM_exp[:, 0], b1)))
    DCM_exp[:, 2] = np.cross(DCM_exp[:, 0], DCM_exp[:, 1]) / np.linalg.norm(
        np.cross(DCM_exp[:, 0], DCM_exp[:, 1]))

    # Fill in the variables for the test
    dcm = sunlineSEKF.new_doubleArray(3 * 3)

    for j in range(9):
        sunlineSEKF.doubleArray_setitem(dcm, j, dcm_BS[j])

    sunlineSEKF.sunlineSEKFComputeDCM_BS(sunheading, bvec1, dcm)

    switchBSout = []
    dcmOut = []
    for j in range(9):
        dcmOut.append(sunlineSEKF.doubleArray_getitem(dcm, j))


    errorNorm = np.zeros(1)
    errorNorm[0] = np.linalg.norm(DCM_exp - np.array(dcmOut).reshape([3, 3]))

    for i in range(len(errorNorm)):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("Frame switch failure \n")

    ###################################################################################
    ## Test the Switching method
    ###################################################################################

    inputStates = [2,1,0.75,0.1,0.4]
    bvec1 = [0.,1.,0.]
    b1 = np.array(bvec1)
    covar = [1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.,
             0., 0., 1., 0., 0.,
             1., 0., 0., 1., 0.,
             0., 1., 0., 0., 1.]
    noise =0.01

    # Fill in expected values for test

    DCM_BSold = np.zeros([3,3])
    DCM_BSnew = np.zeros([3,3])
    Switch = np.eye(numStates)
    SwitchBSold = np.eye(numStates)
    SwitchBSnew = np.eye(numStates)

    DCM_BSold[:,0] = np.array(inputStates[0:3])/(np.linalg.norm(np.array(inputStates[0:3])))
    DCM_BSold[:,1] = np.cross(DCM_BSold[:,0], b1)/np.linalg.norm(np.array(np.cross(DCM_BSold[:,0], b1)))
    DCM_BSold[:,2] = np.cross(DCM_BSold[:,0], DCM_BSold[:,1])/np.linalg.norm(np.cross(DCM_BSold[:,0], DCM_BSold[:,1]))
    SwitchBSold[3:5, 3:5] = DCM_BSold[1:3, 1:3]

    b2 = np.array([1.,0.,0.])
    DCM_BSnew[:,0] = np.array(inputStates[0:3])/(np.linalg.norm(np.array(inputStates[0:3])))
    DCM_BSnew[:,1] = np.cross(DCM_BSnew[:,0], b2)/np.linalg.norm(np.array(np.cross(DCM_BSnew[:,0], b2)))
    DCM_BSnew[:,2] = np.cross(DCM_BSnew[:,0], DCM_BSnew[:,1])/np.linalg.norm(np.cross(DCM_BSnew[:,0], DCM_BSnew[:,1]))
    SwitchBSnew[3:5, 3:5] = DCM_BSnew[1:3, 1:3]

    DCM_newOld = np.dot(DCM_BSnew.T, DCM_BSold)
    Switch[3:5, 3:5] = DCM_newOld[1:3,1:3]

    # Fill in the variables for the test
    bvec = sunlineSEKF.new_doubleArray(3)
    states = sunlineSEKF.new_doubleArray(numStates)
    covarMat = sunlineSEKF.new_doubleArray(numStates * numStates)
    # switchBS = sunlineSEKF.new_doubleArray(numStates * numStates)

    for i in range(3):
        sunlineSEKF.doubleArray_setitem(bvec, i, bvec1[i])
    for i in range(numStates):
        sunlineSEKF.doubleArray_setitem(states, i, inputStates[i])
    for j in range(numStates*numStates):
        sunlineSEKF.doubleArray_setitem(covarMat, j, covar[j])
        # sunlineSEKF.doubleArray_setitem(switchBS, j, switchInput[j])

    sunlineSEKF.sunlineSEKFSwitch(bvec, states, covarMat)

    switchBSout = []
    covarOut = []
    stateOut = []
    bvecOut = []
    for i in range(3):
        bvecOut.append(sunlineSEKF.doubleArray_getitem(bvec, i))
    for i in range(numStates):
        stateOut.append(sunlineSEKF.doubleArray_getitem(states, i))
    for j in range(numStates*numStates):
        covarOut.append(sunlineSEKF.doubleArray_getitem(covarMat, j))


    expectedState = np.dot(Switch, np.array(inputStates))
    Pk = np.array(covar).reshape([numStates, numStates])
    expectedP = np.dot(Switch, np.dot(Pk, Switch.T))

    errorNorm = np.zeros(3)
    errorNorm[0] = np.linalg.norm(np.array(stateOut) - expectedState)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([numStates, numStates]))
    errorNorm[2] = np.linalg.norm(np.array(bvecOut) - b2)
    # errorNorm[3] = np.linalg.norm(SwitchBSnew - np.array(switchBSout).reshape([numStates, numStates]))

    for i in range(len(errorNorm)):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("Frame switch failure \n")


    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + " SEKF individual tests")
    else:
        print(str(testFailCount) + ' tests failed')
        print(testMessages)
    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

####################################################################################
# Test for the time and update with static states (zero d_dot)
####################################################################################
def StatePropStatic():
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True
    numStates = 5
    numObs = 3

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = sunlineSEKF.sunlineSEKF()
    module.ModelTag = "sunlineSEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)

    kfLog = module.logger(["covar", "state"], testProcessRate*10)
    unitTestSim.AddModelToTask(unitTaskName, kfLog)

    # connect messages
    cssDataInMsg = messaging.CSSArraySensorMsg()
    cssConfigInMsg = messaging.CSSConfigMsg()
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConfigInMsg)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(8000.0))
    unitTestSim.ExecuteSimulation()

    stateLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.state)

    for i in range(numStates):
        if (abs(stateLog[-1, i + 1] - stateLog[0, i + 1]) > 1.0E-10):
            testFailCount += 1
            testMessages.append("State propagation failure \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "SEKF static state propagation")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

####################################################################################
# Test for the time and update with changing states (non-zero d_dot)
####################################################################################
def StatePropVariable(show_plots):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    numStates = 5
    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = sunlineSEKF.sunlineSEKF()
    module.ModelTag = "sunlineSEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)

    InitialState =  (np.array(module.state)+ +np.array([0.,0.,0.,0.0001,0.002])).tolist()
    Initialx = module.x
    InitialCovar = module.covar
    module.state = InitialState

    kfLog = module.logger(["covar", "stateTransition", "state", "x"], testProcessRate)
    unitTestSim.AddModelToTask(unitTaskName, kfLog)

    # connect messages
    cssDataInMsg = messaging.CSSArraySensorMsg()
    cssConfigInMsg = messaging.CSSConfigMsg()
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConfigInMsg)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(1000.0))
    unitTestSim.ExecuteSimulation()


    covarLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.covar)
    stateLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.state)
    stateErrorLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.x)
    stmLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.stateTransition)

    bVec = [1.,0.,0.]
    dt = 0.5
    expectedStateArray = np.zeros([2001,numStates+1])
    DCM_BS = np.zeros([2001,3,3])
    omega_S = np.zeros([2001,3])
    omega_B = np.zeros([2001,3])
    expectedStateArray[0,1:numStates+1] = np.array(InitialState)
    expDynMat = np.zeros([2001,numStates,numStates])

    DCM_BS[0,:,0] = np.array(InitialState[0:3])/(np.linalg.norm(np.array(InitialState[0:3])))
    DCM_BS[0,:,1] = np.cross(DCM_BS[0,:,0], bVec)/np.linalg.norm(np.array(np.cross(DCM_BS[0,:,0], bVec)))
    DCM_BS[0,:,2] = np.cross(DCM_BS[0,:,0], DCM_BS[0,:,1])/np.linalg.norm(np.cross(DCM_BS[0,:,0], DCM_BS[0,:,1]))
    omega_S[0,1:] = InitialState[3:]
    omega_B[0,:] = np.dot(DCM_BS[0, :, :], omega_S[0,:])

    for i in range(1,2001):
        expectedStateArray[i,0] = dt*i*1E9
        expectedStateArray[i,1:4] = expectedStateArray[i-1,1:4] + dt * np.cross(omega_B[i-1,:],
                                                                                expectedStateArray[i - 1, 1:4])
        expectedStateArray[i, 4:6] = expectedStateArray[i-1, 4:6]

        # Fill in the variables for the test
        dcm = sunlineSEKF.new_doubleArray(3 * 3)
        for j in range(9):
            sunlineSEKF.doubleArray_setitem(dcm, j, 0)
        sunlineSEKF.sunlineSEKFComputeDCM_BS(expectedStateArray[i, 1:4], bVec, dcm)
        dcmOut = []
        for j in range(9):
            dcmOut.append(sunlineSEKF.doubleArray_getitem(dcm, j))
        DCM_BS[i,:,:] = np.array(dcmOut).reshape([3, 3])
        omega_S[i, 1:] = expectedStateArray[i, 4:]
        omega_B[i,:] = np.dot(DCM_BS[i, :, :], omega_S[i,:])

    for i in range(0, 2001):
        dtilde = -np.array(RigidBodyKinematics.v3Tilde(expectedStateArray[i, 1:4]))
        dBS = np.dot(dtilde, DCM_BS[i,:,:])

        expDynMat[i,0:3, 0:3] = np.array(RigidBodyKinematics.v3Tilde(omega_B[i,:]))
        expDynMat[i, 0:3, 3:numStates] = dBS[:, 1:]
    expectedSTM = np.zeros([2001,numStates,numStates])
    expectedSTM[0,:,:] = np.eye(numStates)
    for i in range(1,2001):
        expectedSTM[i,:,:] = dt * np.dot(expDynMat[i-1,:,:], np.eye(numStates)) + np.eye(numStates)

    expectedXBar = np.zeros([2001,numStates+1])
    expectedXBar[0,1:6] = np.array(Initialx)
    for i in range(1,2001):
        expectedXBar[i,0] = dt*i*1E9
        expectedXBar[i, 1:6] = np.dot(expectedSTM[i, :, :], expectedXBar[i - 1, 1:6])

    expectedCovar = np.zeros([2001,26])
    expectedCovar[0,1:26] = np.array(InitialCovar)
    Gamma = np.zeros([2001,numStates, 2])
    ProcNoiseCovar = np.zeros([2001,numStates,numStates])
    for i in range(0,2001):
        s_skew = np.array([[0., -expectedStateArray[i,3], expectedStateArray[i,2]],
                           [expectedStateArray[i,3], 0., -expectedStateArray[i,1]],
                           [-expectedStateArray[i,2], expectedStateArray[i,1], 0.]])
        s_BS = np.dot(s_skew, DCM_BS[i,:,:])
        Gamma[i, 0:3, 0:2] = dt ** 2. / 2. * s_BS[:,1:3]
        Gamma[i,3:numStates, 0:2] = dt * np.eye(2)
        ProcNoiseCovar[i,:,:] = np.dot(Gamma[i,:,:], np.dot(module.qProcVal*np.eye(2),Gamma[i,:,:].T))
    for i in range(1,2001):
        expectedCovar[i,0] =  dt*i*1E9
        expectedCovar[i,1:26] = (np.dot(expectedSTM[i,:,:], np.dot(np.reshape(expectedCovar[i-1,1:26],[numStates,numStates]), np.transpose(expectedSTM[i,:,:])))+ ProcNoiseCovar[i,:,:]).flatten()
    FilterPlots.StatesVsExpected(stateLog, expectedStateArray, show_plots)
    FilterPlots.StatesPlotCompare(stateErrorLog, expectedXBar, covarLog, expectedCovar, show_plots)

    if (np.linalg.norm(np.array(stateLog)[:, 1:] - expectedStateArray[:, 1:]) > 1.0E-10):
        testFailCount += 1
        testMessages.append("General state propagation failure: State Prop \n")

    if (np.linalg.norm(np.array(stateErrorLog)[:, 1:] - expectedXBar[:,1:]) > 1.0E-4):
        testFailCount += 1
        testMessages.append("General state propagation failure: State Error Prop \n")

    if (np.linalg.norm(np.array(covarLog)[:, 1:] - expectedCovar[:, 1:]) > 1.0E-4):
        testFailCount += 1
        testMessages.append("General state propagation failure: Covariance Prop \n")
    if (np.linalg.norm(np.array(stmLog)[:, 1:] - expectedSTM[:,:,:].reshape([2001,25])) > 1.0E-4):
        testFailCount += 1
        testMessages.append("General state propagation failure: STM Prop \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "SEKF general state propagation")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


####################################################################################
# Test for the full filter with time and measurement update
####################################################################################
def StateUpdateSunLine(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    numStates = 5
    numObs = 3
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = sunlineSEKF.sunlineSEKF()
    module.ModelTag = "sunlineSEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)
    setupFilterData(module)

    # Set up some test parameters

    cssConstelation = messaging.CSSConfigMsgPayload()

    CSSOrientationList = [
        [0.70710678118654746, -0.5, 0.5],
        [0.70710678118654746, -0.5, -0.5],
        [0.70710678118654746, 0.5, -0.5],
        [0.70710678118654746, 0.5, 0.5],
        [-0.70710678118654746, 0, 0.70710678118654757],
        [-0.70710678118654746, 0.70710678118654757, 0.0],
        [-0.70710678118654746, 0, -0.70710678118654757],
        [-0.70710678118654746, -0.70710678118654757, 0.0],
    ]
    CSSBias = [1 for i in range(len(CSSOrientationList))]

    totalCSSList = []
    # Initializing a 2D double array is hard with SWIG.  That's why there is this
    # layer between the above list and the actual C variables.
    i = 0
    for CSSHat in CSSOrientationList:
        newCSS = messaging.CSSUnitConfigMsgPayload()
        newCSS.CBias = CSSBias[i]
        newCSS.nHat_B = CSSHat
        totalCSSList.append(newCSS)
        i = i + 1
    cssConstelation.nCSS = len(CSSOrientationList)
    cssConstelation.cssVals = totalCSSList

    inputData = messaging.CSSArraySensorMsgPayload()

    cssConstInMsg = messaging.CSSConfigMsg().write(cssConstelation)
    cssDataInMsg = messaging.CSSArraySensorMsg()

    # connect messages
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConstInMsg)

    stateTarget1 = testVector1
    stateTarget1 += [0.0, 0.0]
    module.state = stateGuess
    module.x = (np.array(stateTarget1) - np.array(stateGuess)).tolist()

    kfLog = module.logger("x", testProcessRate)
    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)
    unitTestSim.AddModelToTask(unitTaskName, kfLog)

    unitTestSim.InitializeSimulation()

    for i in range(SimHalfLength):
        if i > 20:
            dotList = []
            for element in CSSOrientationList:
                if AddMeasNoise:
                    dotProd = np.dot(np.array(element), np.array(testVector1)[0:3]) + np.random.normal(0., module.qObsVal)
                else:
                    dotProd = np.dot(np.array(element), np.array(testVector1)[0:3])
                dotList.append(dotProd)
            inputData.CosValue = dotList
            cssDataInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)

        unitTestSim.ConfigureStopTime(macros.sec2nano((i + 1) * 0.5))
        unitTestSim.ExecuteSimulation()

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    for i in range(numStates):
        if (abs(covarLog[-1, i *numStates  + 1 + i] - covarLog[0, i * numStates + 1 + i] / 100.) > 1E-1):
            print(abs(covarLog[-1, i *numStates  + 1 + i] - covarLog[0, i * numStates + 1 + i] / 100.))
            testFailCount += 1
            testMessages.append("Covariance update failure")
        if (abs(stateLog[-1, i + 1] - stateTarget1[i]) > 1.0E-1):
            testFailCount += 1
            testMessages.append("State update failure")


    stateTarget2 = testVector2
    stateTarget2 = stateTarget2+[0.,0.]

    inputData = messaging.CSSArraySensorMsgPayload()
    for i in range(SimHalfLength):
        if i > 20:
            dotList = []
            for element in CSSOrientationList:
                if AddMeasNoise:
                    dotProd = np.dot(np.array(element), np.array(testVector2)[0:3])  + np.random.normal(0., module.qObsVal)
                else:
                    dotProd = np.dot(np.array(element), np.array(testVector2)[0:3])
                dotList.append(dotProd)
            inputData.CosValue = dotList
            cssDataInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)

        unitTestSim.ConfigureStopTime(macros.sec2nano((i + SimHalfLength+1) * 0.5))
        unitTestSim.ExecuteSimulation()

    stateErrorLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.x)
    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    postFitLog = addTimeColumn(dataLog.times(), dataLog.postFitRes)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)



    for i in range(numStates):
        if (abs(covarLog[-1, i * numStates + 1 + i] - covarLog[0, i * numStates + 1 + i] / 100.) > 1E-1):
            testFailCount += 1
            testMessages.append("Covariance update failure at end")
        if (abs(stateLog[-1, i + 1] - stateTarget2[i]) > 1.0E-1):
            testFailCount += 1
            testMessages.append("State update failure at end")

    target1 = np.array(testVector1)
    target2 = np.array(testVector2+[0.,0.])
    FilterPlots.StatesPlot(stateErrorLog, covarLog, show_plots)
    FilterPlots.StatesVsTargets(target1, target2, stateLog, show_plots)
    FilterPlots.PostFitResiduals(postFitLog, module.qObsVal, show_plots)

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "SEKF full test")
    else:
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]



if __name__ == "__main__":
    # StatePropVariable(True)
    # sunline_individual_test()
    test_all_sunline_sekf(True, 200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0])
