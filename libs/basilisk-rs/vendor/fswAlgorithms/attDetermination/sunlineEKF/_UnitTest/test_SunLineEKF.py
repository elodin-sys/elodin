#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

#   This test validates the EKF module by running several
#   scenarios on both individual functions and the full module.
#   Author: Thibaud Teil

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import sunlineEKF
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions

import SunLineEKF_test_utilities as FilterPlots


def addTimeColumn(time, data):
    return np.transpose(np.vstack([[time], np.transpose(data)]))


def setupFilterData(filterObject):

    filterObject.sensorUseThresh = 0.
    filterObject.state = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    filterObject.x = [1.0, 0.0, 1.0, 0.0, 0.1, 0.0]
    filterObject.covar = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.4, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.4, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.004, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.004, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.004]

    filterObject.qProcVal = 0.1**2
    filterObject.qObsVal = 0.001
    filterObject.eKFSwitch = 5. #If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF

def test_all_functions_ekf(show_plots):
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
    (200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]),
    (2000, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]),
    (200, False ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]),
    (200, False ,[0., 0.4, -0.4] ,[0., 0.7, 0.2], [0.3, 0.0, 0.6, 0.0, 0.0, 0.0]),
    (200, True ,[0., 0.4, -0.4] ,[0.4, 0.5, 0.], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]),
    (200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0])
])


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def test_all_sunline_ekf(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess):
    """Module Unit Test"""
    [testResults, testMessage] = StateUpdateSunLine(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess)
    assert testResults < 1, testMessage


def sunline_individual_test():
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    ###################################################################################
    ## Testing dynamics matrix computation
    ###################################################################################

    inputStates = [2,1,0.75,0.1,0.4,0.05]
    dt =0.5

    expDynMat = np.zeros([6,6])
    expDynMat[0:3, 0:3] = -(np.outer(inputStates[0:3],inputStates[3:6])/np.linalg.norm(inputStates[0:3])**2. +
                         np.dot(inputStates[3:6],inputStates[0:3])*(np.linalg.norm(inputStates[0:3])**2.*np.eye(3)- 2*np.outer(inputStates[0:3],inputStates[0:3]))/np.linalg.norm(inputStates[0:3])**4.)
    expDynMat[0:3, 3:6] = np.eye(3) - np.outer(inputStates[0:3],inputStates[0:3])/np.linalg.norm(inputStates[0:3])**2

    ## Equations when removing the unobservable states from d_dot
    expDynMat[3:6, 0:3] = -1/dt*(np.outer(inputStates[0:3],inputStates[3:6])/np.linalg.norm(inputStates[0:3])**2. +
                         np.dot(inputStates[3:6],inputStates[0:3])*(np.linalg.norm(inputStates[0:3])**2.*np.eye(3)- 2*np.outer(inputStates[0:3],inputStates[0:3]))/np.linalg.norm(inputStates[0:3])**4.)
    expDynMat[3:6, 3:6] =- 1/dt*(np.outer(inputStates[0:3],inputStates[0:3])/np.linalg.norm(inputStates[0:3])**2)

    dynMat = sunlineEKF.new_doubleArray(6*6)
    for i in range(36):
        sunlineEKF.doubleArray_setitem(dynMat, i, 0.0)
    sunlineEKF.sunlineDynMatrix(inputStates, dt, dynMat)

    DynOut = []
    for i in range(36):
        DynOut.append(sunlineEKF.doubleArray_getitem(dynMat, i))

    DynOut = np.array(DynOut).reshape(6, 6)
    errorNorm = np.linalg.norm(expDynMat - DynOut)
    if(errorNorm > 1.0E-10):
        print(errorNorm)
        testFailCount += 1
        testMessages.append("Dynamics Matrix generation Failure \n")

    ###################################################################################
    ## STM and State Test
    ###################################################################################

    inputStates = [2,1,0.75, 1.5, 0.5, 0.5]
    dt =0.5
    stateTransition = sunlineEKF.new_doubleArray(36)
    states = sunlineEKF.new_doubleArray(6)
    for i in range(6):
        sunlineEKF.doubleArray_setitem(states, i, inputStates[i])
        for j in range(6):
            if i==j:
                sunlineEKF.doubleArray_setitem(stateTransition, 6*i+j, 1.0)
            else:
                sunlineEKF.doubleArray_setitem(stateTransition, 6*i+j, 0.0)

    sunlineEKF.sunlineStateSTMProp(expDynMat.flatten().tolist(), dt, states, stateTransition)

    PropStateOut = []
    PropSTMOut = []
    for i in range(6):
        PropStateOut.append(sunlineEKF.doubleArray_getitem(states, i))
    for i in range(36):
        PropSTMOut.append(sunlineEKF.doubleArray_getitem(stateTransition, i))

    STMout = np.array(PropSTMOut).reshape([6,6])
    StatesOut = np.array(PropStateOut)

    expectedSTM = dt*np.dot(expDynMat, np.eye(6)) + np.eye(6)
    expectedStates = np.zeros(6)
    inputStatesArray = np.array(inputStates)
    ## Equations when removing the unobservable states from d_dot
    expectedStates[3:6] = np.array(inputStatesArray[3:6]  - np.dot(inputStatesArray[3:6], inputStatesArray[0:3])*inputStatesArray[0:3]/np.linalg.norm(inputStatesArray[0:3])**2.)
    expectedStates[0:3] = np.array(inputStatesArray[0:3] + dt*(inputStatesArray[3:6] - np.dot(inputStatesArray[3:6], inputStatesArray[0:3])*inputStatesArray[0:3]/np.linalg.norm(inputStatesArray[0:3])**2.))
    errorNormSTM = np.linalg.norm(expectedSTM - STMout)
    errorNormStates = np.linalg.norm(expectedStates - StatesOut)

    if(errorNormSTM > 1.0E-10):
        testFailCount += 1
        testMessages.append("STM Propagation Failure \n")

    if(errorNormStates > 1.0E-10):
        testFailCount += 1
        testMessages.append("State Propagation Failure \n")

    ###################################################################################
    ## Test the H and yMeas matrix generation as well as the observation count
    ###################################################################################

    numCSS = 4
    cssCos = [np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)), np.cos(np.deg2rad(90.))]
    sensorTresh = np.cos(np.deg2rad(50.))
    cssNormals = [1.,0.,0.,0.,1.,0., 0.,0.,1., 1./np.sqrt(2), 1./np.sqrt(2),0.]
    cssBias = [1.0 for i in range(numCSS)]

    measMat = sunlineEKF.new_doubleArray(8*6)
    obs = sunlineEKF.new_doubleArray(8)
    yMeas = sunlineEKF.new_doubleArray(8)
    numObs = sunlineEKF.new_intArray(1)

    for i in range(8*6):
        sunlineEKF.doubleArray_setitem(measMat, i, 0.)
    for i in range(8):
        sunlineEKF.doubleArray_setitem(obs, i, 0.0)
        sunlineEKF.doubleArray_setitem(yMeas, i, 0.0)

    sunlineEKF.sunlineHMatrixYMeas(inputStates, numCSS, cssCos, sensorTresh, cssNormals, cssBias, obs, yMeas, numObs, measMat)

    obsOut = []
    yMeasOut = []
    numObsOut = []
    HOut = []
    for i in range(8*6):
        HOut.append(sunlineEKF.doubleArray_getitem(measMat, i))
    for i in range(8):
        yMeasOut.append(sunlineEKF.doubleArray_getitem(yMeas, i))
        obsOut.append(sunlineEKF.doubleArray_getitem(obs, i))
    numObsOut.append(sunlineEKF.intArray_getitem(numObs, 0))

    #Fill in expected values for test
    expectedH = np.zeros([8,6])
    expectedY = np.zeros(8)
    for j in range(3):
        expectedH[j,0:3] = np.eye(3)[j,:]
        expectedY[j] =np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3], np.array(cssNormals)[j*3:(j+1)*3])
    expectedObs = np.array([np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)),0.,0.,0.,0.,0.])
    expectedNumObs = 3

    HOut = np.array(HOut).reshape([8, 6])
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
    h = [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0., 0.,
        0., 1., 0., 0., 1., 0.,
        0., 0., 1., 0., 0., 1.,
        1., 0., 0., 1., 0., 0.,
        0., 1., 0., 0., 1., 0.,
        0., 0., 1., 0., 0., 1.]
    noise= 0.01

    Kalman = sunlineEKF.new_doubleArray(6 * 8)

    for i in range(8 * 6):
        sunlineEKF.doubleArray_setitem(Kalman, i, 0.)

    sunlineEKF.sunlineKalmanGain(covar, h, noise, numObs, Kalman)

    KalmanOut = []
    for i in range(8 * 6):
        KalmanOut.append(sunlineEKF.doubleArray_getitem(Kalman, i))

    # Fill in expected values for test
    Hmat = np.array(h).reshape([8,6])
    Pk = np.array(covar).reshape([6,6])
    R = noise*np.eye(3)
    expectedK = np.dot(np.dot(Pk, Hmat[0:numObs,:].T), np.linalg.inv(np.dot(np.dot(Hmat[0:numObs,:], Pk), Hmat[0:numObs,:].T) + R[0:numObs,0:numObs]))

    KalmanOut = np.array(KalmanOut)[0:6*numObs].reshape([6, 3])
    errorNorm = np.linalg.norm(KalmanOut[:,0:numObs] - expectedK)

    if (errorNorm > 1.0E-10):
        print(errorNorm)
        testFailCount += 1
        testMessages.append("Kalman Gain update failure \n")

    ###################################################################################
    ## Test the EKF update
    ###################################################################################

    KGain = [1.,2.,3., 0., 1., 2., 1., 0., 1., 0., 1., 0., 3., 0., 1., 0., 2., 0.]
    for i in range(6*8-6*3):
        KGain.append(0.)
    inputStates = [2,1,0.75,0.1,0.4,0.05]
    xbar = [0.1, 0.2, 0.01, 0.005, 0.009, 0.001]
    numObs = 3
    h = [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.,
             1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.]
    noise = 0.01
    inputY = np.zeros(3)
    for j in range(3):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3], np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = sunlineEKF.new_doubleArray(6)
    covarMat = sunlineEKF.new_doubleArray(6*6)
    inputs = sunlineEKF.new_doubleArray(6)


    for i in range(6):
        sunlineEKF.doubleArray_setitem(stateError, i, 0.)
        sunlineEKF.doubleArray_setitem(inputs, i, inputStates[i])
        for j in range(6):
            sunlineEKF.doubleArray_setitem(covarMat,i+j,0.)

    sunlineEKF.sunlineEKFUpdate(KGain, covar, noise, numObs, inputY, h, inputs, stateError, covarMat)

    stateOut = []
    covarOut = []
    errorOut = []
    for i in range(6):
        stateOut.append(sunlineEKF.doubleArray_getitem(inputs, i))
        errorOut.append(sunlineEKF.doubleArray_getitem(stateError, i))
    for j in range(36):
        covarOut.append(sunlineEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:6*3].reshape([6,3])
    expectedStates = np.array(inputStates) + np.dot(KK, np.array(inputY))
    H = np.array(h).reshape([8,6])[0:3,:]
    Pk = np.array(covar).reshape([6, 6])
    R = noise * np.eye(3)
    expectedP = np.dot(np.dot(np.eye(6) - np.dot(KK, H), Pk), np.transpose(np.eye(6) - np.dot(KK, H))) + np.dot(KK, np.dot(R,KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(stateOut) - expectedStates)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([6,6]))

    for i in range(2):
        if(errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("EKF update failure \n")

    ###################################################################################
    ## Test the CKF update
    ###################################################################################

    KGain = [1., 2., 3., 0., 1., 2., 1., 0., 1., 0., 1., 0., 3., 0., 1., 0., 2., 0.]
    for i in range(6 * 8 - 6 * 3):
        KGain.append(0.)
    inputStates = [2, 1, 0.75, 0.1, 0.4, 0.05]
    xbar = [0.1, 0.2, 0.01, 0.005, 0.009, 0.001]
    numObs = 3
    h = [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.,
             1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.]
    noise =0.01
    inputY = np.zeros(3)
    for j in range(3):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3],
                                                 np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = sunlineEKF.new_doubleArray(6)
    covarMat = sunlineEKF.new_doubleArray(6 * 6)

    for i in range(6):
        sunlineEKF.doubleArray_setitem(stateError, i, xbar[i])
        for j in range(6):
            sunlineEKF.doubleArray_setitem(covarMat, i + j, 0.)

    sunlineEKF.sunlineCKFUpdate(xbar, KGain, covar, noise, numObs, inputY, h, stateError, covarMat)

    covarOut = []
    errorOut = []
    for i in range(6):
        errorOut.append(sunlineEKF.doubleArray_getitem(stateError, i))
    for j in range(36):
        covarOut.append(sunlineEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:6 * 3].reshape([6, 3])
    H = np.array(h).reshape([8, 6])[0:3, :]
    expectedStateError = np.array(xbar) + np.dot(KK, (np.array(inputY) - np.dot(H, np.array(xbar))))
    Pk = np.array(covar).reshape([6, 6])
    expectedP = np.dot(np.dot(np.eye(6) - np.dot(KK, H), Pk), np.transpose(np.eye(6) - np.dot(KK, H))) + np.dot(KK,
                                                                                                                np.dot(
                                                                                                                    R,
                                                                                                                    KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(errorOut) - expectedStateError)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([6, 6]))

    for i in range(2):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("CKF update failure \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + " EKF individual tests")
    else:
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
    module = sunlineEKF.sunlineEKF()
    module.ModelTag = "SunlineEKF"

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

    for i in range(6):
        if (abs(stateLog[-1, i + 1] - stateLog[0, i + 1]) > 1.0E-10):
            testFailCount += 1
            testMessages.append("State propagation failure \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "EKF static state propagation")
    else:
        print(testMessages)

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

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = sunlineEKF.sunlineEKF()
    module.ModelTag = "SunlineEKF"



    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)

    InitialState = (np.array(module.state)+ +np.array([0.,0.,0.,0.0001,0.002, 0.001])).tolist()
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


    dt = 0.5
    expectedStateArray = np.zeros([2001,7])
    expectedStateArray[0,1:7] = np.array(InitialState)

    for i in range(1,2001):
        expectedStateArray[i,0] = dt*i*1E9
        expectedStateArray[i,1:4] = expectedStateArray[i-1,1:4] + dt*(expectedStateArray[i-1,4:7] - (np.dot(expectedStateArray[i-1,4:7],expectedStateArray[i-1,1:4]))*expectedStateArray[i-1,1:4]/np.linalg.norm(expectedStateArray[i-1,1:4])**2.)
        ## Equations when removing the unobservable states from d_dot
        expectedStateArray[i, 4:7] = expectedStateArray[i-1,4:7] - (np.dot(expectedStateArray[i-1,4:7],expectedStateArray[i-1,1:4]))*expectedStateArray[i-1,1:4]/np.linalg.norm(expectedStateArray[i-1,1:4])**2.

    expDynMat = np.zeros([2001,6,6])
    for i in range(0,2001):
        expDynMat[i, 0:3, 0:3] = -(np.outer(expectedStateArray[i,1:4],expectedStateArray[i,4:7])/np.linalg.norm(expectedStateArray[i,1:4])**2. +
                             np.dot(expectedStateArray[i,4:7], expectedStateArray[i,1:4])*(np.linalg.norm(expectedStateArray[i,1:4])**2.*np.eye(3)- 2*np.outer(expectedStateArray[i,1:4],expectedStateArray[i,1:4]))/np.linalg.norm(expectedStateArray[i,1:4])**4.)
        expDynMat[i, 0:3, 3:6] = np.eye(3) - np.outer(expectedStateArray[i,1:4],expectedStateArray[i,1:4])/np.linalg.norm(expectedStateArray[i,1:4])**2
        ## Equations when removing the unobservable states from d_dot
        expDynMat[i, 3:6, 0:3] = -1/dt*(np.outer(expectedStateArray[i,1:4],expectedStateArray[i,4:7])/np.linalg.norm(expectedStateArray[i,1:4])**2. +
                             np.dot(expectedStateArray[i,4:7], expectedStateArray[i,1:4])*(np.linalg.norm(expectedStateArray[i,1:4])**2.*np.eye(3)- 2*np.outer(expectedStateArray[i,1:4],expectedStateArray[i,1:4]))/np.linalg.norm(expectedStateArray[i,1:4])**4.)
        expDynMat[i, 3:6, 3:6] = -1/dt*(np.outer(expectedStateArray[i,1:4],expectedStateArray[i,1:4])/np.linalg.norm(expectedStateArray[i,1:4])**2)

    expectedSTM = np.zeros([2001,6,6])
    expectedSTM[0,:,:] = np.eye(6)
    for i in range(1,2001):
        expectedSTM[i,:,:] = dt * np.dot(expDynMat[i-1,:,:], np.eye(6)) + np.eye(6)

    expectedXBar = np.zeros([2001,7])
    expectedXBar[0,1:7] = np.array(Initialx)
    for i in range(1,2001):
        expectedXBar[i,0] = dt*i*1E9
        expectedXBar[i, 1:7] = np.dot(expectedSTM[i, :, :], expectedXBar[i - 1, 1:7])

    expectedCovar = np.zeros([2001,37])
    expectedCovar[0,1:37] = np.array(InitialCovar)
    Gamma = np.zeros([6, 3])
    Gamma[0:3, 0:3] = dt ** 2. / 2. * np.eye(3)
    Gamma[3:6, 0:3] = dt * np.eye(3)
    ProcNoiseCovar = np.dot(Gamma, np.dot(module.qProcVal*np.eye(3),Gamma.T))
    for i in range(1,2001):
        expectedCovar[i,0] =  dt*i*1E9
        expectedCovar[i,1:37] = (np.dot(expectedSTM[i,:,:], np.dot(np.reshape(expectedCovar[i-1,1:37],[6,6]), np.transpose(expectedSTM[i,:,:])))+ ProcNoiseCovar).flatten()

    FilterPlots.StatesVsExpected(stateLog, expectedStateArray, show_plots)
    FilterPlots.StatesPlotCompare(stateErrorLog, expectedXBar, covarLog, expectedCovar, show_plots)

    for j in range(1,2001):
        for i in range(6):
            if (abs(stateLog[j, i + 1] - expectedStateArray[j, i + 1]) > 1.0E-4):
                testFailCount += 1
                testMessages.append("General state propagation failure: State Prop \n")
            if (abs(stateErrorLog[j, i + 1] - expectedXBar[j, i + 1]) > 1.0E-4):
                testFailCount += 1
                testMessages.append("General state propagation failure: State Error Prop \n")

        for i in range(36):
            if (abs(covarLog[j, i + 1] - expectedCovar[j, i + 1]) > 1.0E-4):
                abs(covarLog[j, i + 1] - expectedCovar[j, i + 1])
                testFailCount += 1
                testMessages.append("General state propagation failure: Covariance Prop \n")
            if (abs(stmLog[j, i + 1] - expectedSTM[j,:].flatten()[i]) > 1.0E-4):
                testFailCount += 1
                testMessages.append("General state propagation failure: STM Prop \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "EKF general state propagation")
    else:
        print(testMessages)

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

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = sunlineEKF.sunlineEKF()
    module.ModelTag = "SunlineEKF"

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
        i = i+1
    cssConstelation.nCSS = len(CSSOrientationList)
    cssConstelation.cssVals = totalCSSList

    inputData = messaging.CSSArraySensorMsgPayload()

    cssConstInMsg = messaging.CSSConfigMsg().write(cssConstelation)
    cssDataInMsg = messaging.CSSArraySensorMsg()

    # connect messages
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConstInMsg)

    stateTarget1 = testVector1
    stateTarget1 += [0.0, 0.0, 0.0]
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

    for i in range(6):
        if (abs(covarLog[-1, i * 6 + 1 + i] - covarLog[0, i * 6 + 1 + i] / 100.) > 1E-2):
            testFailCount += 1
            testMessages.append("Covariance update failure")
        if (abs(stateLog[-1, i + 1] - stateTarget1[i]) > 1.0E-2):
            testFailCount += 1
            testMessages.append("State update failure")


    stateTarget2 = testVector2
    stateTarget2 = stateTarget2+[0.,0.,0.]

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

    for i in range(6):
        if (abs(covarLog[-1, i * 6 + 1 + i] - covarLog[0, i * 6 + 1 + i] / 100.) > 1E-2):
            testFailCount += 1
            testMessages.append("Covariance update failure")
        if (abs(stateLog[-1, i + 1] - stateTarget2[i]) > 1.0E-2):
            testFailCount += 1
            testMessages.append("State update failure")

    target1 = np.array(testVector1)
    target2 = np.array(testVector2+[0.,0.,0.])
    FilterPlots.StateErrorCovarPlot(stateErrorLog, covarLog, show_plots)
    FilterPlots.StatesVsTargets(target1, target2, stateLog, show_plots)
    FilterPlots.PostFitResiduals(postFitLog, module.qObsVal, show_plots)

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "EKF full test")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]



if __name__ == "__main__":
    # test_all_sunline_ekf(True, 200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0])
    # StatePropVariable(True)
    # StatePropStatic()
    StateUpdateSunLine(True, 200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0, 0.0, 0.0, 0.0])
