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
import inspect
import os
import sys

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('FswAlgorithms')
sys.path.append(splitPath[0] + '/modules')
sys.path.append(splitPath[0] + '/PythonModules')

import SunLineOEKF_test_utilities as FilterPlots
from Basilisk.fswAlgorithms import okeefeEKF
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from Basilisk.architecture import messaging


def setupFilterData(filterObject):
    filterObject.sensorUseThresh = 0.
    filterObject.state = [1.0, 1.0, 1.0]
    filterObject.omega = [0.1, 0.2, 0.1]
    filterObject.x = [1.0, 0.0, 1.0]
    filterObject.covar = [0.4, 0.0, 0.0,
                          0.0, 0.4, 0.0,
                          0.0, 0.0, 0.4]

    filterObject.qProcVal = 0.1**2
    filterObject.qObsVal = 0.001
    filterObject.eKFSwitch = 5. #If low (0-5), the CKF kicks in easily, if high (>10) it's mostly only EKF

def test_all_functions_oekf(show_plots):
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
    (200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0]),
    (2000, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0]),
    (200, False ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0]),
    (200, False ,[0., 0.4, -0.4] ,[0., 0.7, 0.2], [0.3, 0.0, 0.6]),
    (200, True ,[0., 0.4, -0.4] ,[0.4, 0.5, 0.], [0.7, 0.7, 0.0]),
    (200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0])
])


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def test_all_sunline_oekf(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess):
    [testResults, testMessage] = StateUpdateSunLine(show_plots, SimHalfLength, AddMeasNoise, testVector1, testVector2, stateGuess)
    assert testResults < 1, testMessage


def sunline_individual_test():
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    NUMSTATES = 3
    ###################################################################################
    ## Testing dynamics matrix computation
    ###################################################################################

    inputOmega = [0.1, 0.2, 0.1]
    dt =0.5

    expDynMat = - np.array([[0., -inputOmega[2], inputOmega[1]],
                            [inputOmega[2], 0., -inputOmega[0]],
                            [ -inputOmega[1], inputOmega[0], 0.]])


    dynMat = okeefeEKF.new_doubleArray(3*3)
    for i in range(9):
        okeefeEKF.doubleArray_setitem(dynMat, i, 0.0)
    okeefeEKF.sunlineDynMatrixOkeefe(inputOmega, dt, dynMat)

    DynOut = []
    for i in range(NUMSTATES*NUMSTATES):
        DynOut.append(okeefeEKF.doubleArray_getitem(dynMat, i))

    DynOut = np.array(DynOut).reshape(3, 3)
    errorNorm = np.linalg.norm(expDynMat - DynOut)
    if(errorNorm > 1.0E-12):
        print(errorNorm)
        testFailCount += 1
        testMessages.append("Dynamics Matrix generation Failure \n")


    ###################################################################################
    ## Testing omega computation
    ###################################################################################

    inputStates = [2,1,0.75]
    inputPrevStates = [1,0.1,0.5]
    norm1 = np.linalg.norm(np.array(inputStates))
    norm2 =  np.linalg.norm(np.array(inputPrevStates))
    dt =0.5

    expOmega = 1./dt*np.cross(np.array(inputStates),np.array(inputPrevStates))/(norm1*norm2)*np.arccos(np.dot(np.array(inputStates),np.array(inputPrevStates))/(norm1*norm2))

    omega = okeefeEKF.new_doubleArray(NUMSTATES)
    for i in range(3):
        okeefeEKF.doubleArray_setitem(omega, i, 0.0)
    okeefeEKF.sunlineRateCompute(inputStates, dt, inputPrevStates, omega)

    omegaOut = []
    for i in range(NUMSTATES):
        omegaOut.append(okeefeEKF.doubleArray_getitem(omega, i))

    omegaOut = np.array(omegaOut)
    errorNorm = np.linalg.norm(expOmega - omegaOut)
    if(errorNorm > 1.0E-12):
        print(errorNorm)
        testFailCount += 1
        testMessages.append("Dynamics Matrix generation Failure \n")

    ###################################################################################
    ## STM and State Test
    ###################################################################################

    inputStates = [2,1,0.75]
    inputOmega = [0.1, 0.2, 0.1]
    dt =0.5
    stateTransition = okeefeEKF.new_doubleArray(NUMSTATES*NUMSTATES)
    states = okeefeEKF.new_doubleArray(NUMSTATES)
    prev_states = okeefeEKF.new_doubleArray(NUMSTATES)
    for i in range(NUMSTATES):
        okeefeEKF.doubleArray_setitem(states, i, inputStates[i])
        for j in range(NUMSTATES):
            if i==j:
                okeefeEKF.doubleArray_setitem(stateTransition, NUMSTATES*i+j, 1.0)
            else:
                okeefeEKF.doubleArray_setitem(stateTransition, NUMSTATES*i+j, 0.0)

    okeefeEKF.sunlineStateSTMProp(expDynMat.flatten().tolist(), dt, inputOmega, states, prev_states, stateTransition)

    PropStateOut = []
    PropSTMOut = []
    for i in range(NUMSTATES):
        PropStateOut.append(okeefeEKF.doubleArray_getitem(states, i))
    for i in range(NUMSTATES*NUMSTATES):
        PropSTMOut.append(okeefeEKF.doubleArray_getitem(stateTransition, i))

    STMout = np.array(PropSTMOut).reshape([NUMSTATES,NUMSTATES])
    StatesOut = np.array(PropStateOut)

    expectedSTM = dt*np.dot(expDynMat, np.eye(NUMSTATES)) + np.eye(NUMSTATES)
    expectedStates = np.zeros(NUMSTATES)
    inputStatesArray = np.array(inputStates)
    ## Equations when removing the unobservable states from d_dot
    expectedStates[0:3] = np.array(inputStatesArray - dt*(np.cross(np.array(inputOmega), np.array(inputStatesArray))))
    errorNormSTM = np.linalg.norm(expectedSTM - STMout)
    errorNormStates = np.linalg.norm(expectedStates - StatesOut)

    if(errorNormSTM > 1.0E-12):
        print(errorNormSTM)
        testFailCount += 1
        testMessages.append("STM Propagation Failure \n")


    if(errorNormStates > 1.0E-12):
        print(errorNormStates)
        testFailCount += 1
        testMessages.append("State Propagation Failure \n")



    ###################################################################################
    # ## Test the H and yMeas matrix generation as well as the observation count
    # ###################################################################################
    numCSS = 4
    cssCos = [np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)), np.cos(np.deg2rad(90.))]
    sensorTresh = np.cos(np.deg2rad(50.))
    cssNormals = [1.,0.,0.,0.,1.,0., 0.,0.,1., 1./np.sqrt(2), 1./np.sqrt(2),0.]
    cssBias = [1.0 for i in range(numCSS)]

    measMat = okeefeEKF.new_doubleArray(8*NUMSTATES)
    obs = okeefeEKF.new_doubleArray(8)
    yMeas = okeefeEKF.new_doubleArray(8)
    numObs = okeefeEKF.new_intArray(1)

    for i in range(8*NUMSTATES):
        okeefeEKF.doubleArray_setitem(measMat, i, 0.)
    for i in range(8):
        okeefeEKF.doubleArray_setitem(obs, i, 0.0)
        okeefeEKF.doubleArray_setitem(yMeas, i, 0.0)

    okeefeEKF.sunlineHMatrixYMeas(inputStates, numCSS, cssCos, sensorTresh, cssNormals, cssBias, obs, yMeas, numObs, measMat)

    obsOut = []
    yMeasOut = []
    numObsOut = []
    HOut = []
    for i in range(8*NUMSTATES):
        HOut.append(okeefeEKF.doubleArray_getitem(measMat, i))
    for i in range(8):
        yMeasOut.append(okeefeEKF.doubleArray_getitem(yMeas, i))
        obsOut.append(okeefeEKF.doubleArray_getitem(obs, i))
    numObsOut.append(okeefeEKF.intArray_getitem(numObs, 0))

    #Fill in expected values for test
    expectedH = np.zeros([8,NUMSTATES])
    expectedY = np.zeros(8)
    for j in range(3):
        expectedH[j,0:3] = np.eye(3)[j,:]
        expectedY[j] =np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3], np.array(cssNormals)[j*3:(j+1)*3])
    expectedObs = np.array([np.cos(np.deg2rad(10.)), np.cos(np.deg2rad(25.)), np.cos(np.deg2rad(5.)),0.,0.,0.,0.,0.])
    expectedNumObs = 3

    HOut = np.array(HOut).reshape([8, NUMSTATES])
    errorNorm = np.zeros(4)
    errorNorm[0] = np.linalg.norm(HOut - expectedH)
    errorNorm[1] = np.linalg.norm(yMeasOut - expectedY)
    errorNorm[2] = np.linalg.norm(obsOut - expectedObs)
    errorNorm[3] = np.linalg.norm(numObsOut[0] - expectedNumObs)
    for i in range(4):
        if(errorNorm[i] > 1.0E-12):
            testFailCount += 1
            testMessages.append("H and yMeas update failure \n")

    # ###################################################################################
    # ## Test the Kalman Gain
    # ###################################################################################

    numObs = 3
    h = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 1.,
        0., 1., 0.,
        1., 0., 1. ]
    noise= 0.01

    Kalman = okeefeEKF.new_doubleArray(NUMSTATES * 8)

    for i in range(8 * NUMSTATES):
        okeefeEKF.doubleArray_setitem(Kalman, i, 0.)

    okeefeEKF.sunlineKalmanGainOkeefe(covar, h, noise, numObs, Kalman)

    KalmanOut = []
    for i in range(8 * NUMSTATES):
        KalmanOut.append(okeefeEKF.doubleArray_getitem(Kalman, i))

    # Fill in expected values for test
    Hmat = np.array(h).reshape([8,NUMSTATES])
    Pk = np.array(covar).reshape([NUMSTATES,NUMSTATES])
    R = noise*np.eye(3)
    expectedK = np.dot(np.dot(Pk, Hmat[0:numObs,:].T), np.linalg.inv(np.dot(np.dot(Hmat[0:numObs,:], Pk), Hmat[0:numObs,:].T) + R[0:numObs,0:numObs]))

    KalmanOut = np.array(KalmanOut)[0:NUMSTATES*numObs].reshape([NUMSTATES, 3])
    errorNorm = np.linalg.norm(KalmanOut[:,0:numObs] - expectedK)


    if (errorNorm > 1.0E-12):
        print(errorNorm)
        testFailCount += 1
        testMessages.append("Kalman Gain update failure \n")

    # ###################################################################################
    # ## Test the EKF update
    # ###################################################################################

    KGain = [1.,2.,3., 0., 1., 2., 1., 0., 1.]
    for i in range(NUMSTATES*8-NUMSTATES*3):
        KGain.append(0.)
    inputStates = [2,1,0.75]
    xbar = [0.1, 0.2, 0.01]
    numObs = 3
    h = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 1.,
             0., 1., 0.,
             1., 0., 1.,
             ]
    noise = 0.01
    inputY = np.zeros(3)
    for j in range(3):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3], np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = okeefeEKF.new_doubleArray(NUMSTATES)
    covarMat = okeefeEKF.new_doubleArray(NUMSTATES*NUMSTATES)
    inputs = okeefeEKF.new_doubleArray(NUMSTATES)


    for i in range(NUMSTATES):
        okeefeEKF.doubleArray_setitem(stateError, i, 0.)
        okeefeEKF.doubleArray_setitem(inputs, i, inputStates[i])
        for j in range(NUMSTATES):
            okeefeEKF.doubleArray_setitem(covarMat,i+j,0.)

    okeefeEKF.okeefeEKFUpdate(KGain, covar, noise, numObs, inputY, h, inputs, stateError, covarMat)

    stateOut = []
    covarOut = []
    errorOut = []
    for i in range(NUMSTATES):
        stateOut.append(okeefeEKF.doubleArray_getitem(inputs, i))
        errorOut.append(okeefeEKF.doubleArray_getitem(stateError, i))
    for j in range(NUMSTATES*NUMSTATES):
        covarOut.append(okeefeEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:NUMSTATES*3].reshape([NUMSTATES,3])
    expectedStates = np.array(inputStates) + np.dot(KK, np.array(inputY))
    H = np.array(h).reshape([8,NUMSTATES])[0:3,:]
    Pk = np.array(covar).reshape([NUMSTATES, NUMSTATES])
    R = noise * np.eye(3)
    expectedP = np.dot(np.dot(np.eye(NUMSTATES) - np.dot(KK, H), Pk), np.transpose(np.eye(NUMSTATES) - np.dot(KK, H))) + np.dot(KK, np.dot(R,KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(stateOut) - expectedStates)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([NUMSTATES,NUMSTATES]))
    for i in range(2):
        if(errorNorm[i] > 1.0E-12):
            testFailCount += 1
            testMessages.append("EKF update failure \n")
    #
    # ###################################################################################
    # ## Test the CKF update
    # ###################################################################################

    KGain = [1., 2., 3.]
    for i in range(NUMSTATES * 8 - NUMSTATES * 3):
        KGain.append(0.)
    inputStates = [2, 1, 0.75]
    xbar = [0.1, 0.2, 0.01]
    numObs = 3
    h = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0.]
    covar = [1., 0., 1.,
             0., 1., 0.,
             1., 0., 1.]
    noise =0.01
    inputY = np.zeros(3)
    for j in range(3):
        inputY[j] = np.array(cssCos[j]) - np.dot(np.array(inputStates)[0:3],
                                                 np.array(cssNormals)[j * 3:(j + 1) * 3])
    inputY = inputY.tolist()

    stateError = okeefeEKF.new_doubleArray(NUMSTATES)
    covarMat = okeefeEKF.new_doubleArray(NUMSTATES * NUMSTATES)

    for i in range(NUMSTATES):
        okeefeEKF.doubleArray_setitem(stateError, i, xbar[i])
        for j in range(NUMSTATES):
            okeefeEKF.doubleArray_setitem(covarMat, i + j, 0.)

    okeefeEKF.sunlineCKFUpdateOkeefe(xbar, KGain, covar, noise, numObs, inputY, h, stateError, covarMat)

    covarOut = []
    errorOut = []
    for i in range(NUMSTATES):
        errorOut.append(okeefeEKF.doubleArray_getitem(stateError, i))
    for j in range(NUMSTATES*NUMSTATES):
        covarOut.append(okeefeEKF.doubleArray_getitem(covarMat, j))

    # Fill in expected values for test
    KK = np.array(KGain)[0:NUMSTATES * 3].reshape([NUMSTATES, 3])
    H = np.array(h).reshape([8, NUMSTATES])[0:3, :]
    expectedStateError = np.array(xbar) + np.dot(KK, (np.array(inputY) - np.dot(H, np.array(xbar))))
    Pk = np.array(covar).reshape([NUMSTATES, NUMSTATES])
    expectedP = np.dot(np.dot(np.eye(NUMSTATES) - np.dot(KK, H), Pk), np.transpose(np.eye(NUMSTATES) - np.dot(KK, H))) + np.dot(KK,
                                                                                                                np.dot(
                                                                                                                    R,
                                                                                                                    KK.T))

    errorNorm = np.zeros(2)
    errorNorm[0] = np.linalg.norm(np.array(errorOut) - expectedStateError)
    errorNorm[1] = np.linalg.norm(expectedP - np.array(covarOut).reshape([NUMSTATES, NUMSTATES]))
    for i in range(2):
        if (errorNorm[i] > 1.0E-12):
            testFailCount += 1
            testMessages.append("CKF update failure \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + " EKF individual tests")

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

    NUMSTATES=3
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
    module = okeefeEKF.okeefeEKF()
    module.ModelTag = "okeefeEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)
    module.omega = [0.,0.,0.]

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

    for i in range(NUMSTATES):
        if (abs(stateLog[-1, i + 1] - stateLog[0, i + 1]) > 1.0E-10):
            print(abs(stateLog[-1, i + 1] - stateLog[0, i + 1]))
            testFailCount += 1
            testMessages.append("Static state propagation failure \n")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "EKF static state propagation")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


####################################################################################
# Test for the time and update with changing states non-zero omega
####################################################################################
def StatePropVariable(show_plots):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    NUMSTATES=3

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
    module = okeefeEKF.okeefeEKF()
    module.ModelTag = "okeefeEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)

    InitialState = module.state
    Initialx = module.x
    InitialCovar = module.covar
    InitOmega = module.omega

    module.state = InitialState
    kfLog = module.logger(["covar", "state", "stateTransition", "x", "omega"], testProcessRate)
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
    omegaLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.omega)

    dt = 0.5
    expectedOmega = np.zeros([2001, (NUMSTATES + 1)])
    expectedStateArray = np.zeros([2001,(NUMSTATES+1)])
    expectedPrevArray = np.zeros([2001,(NUMSTATES+1)])
    expectedStateArray[0,1:(NUMSTATES+1)] = np.array(InitialState)
    expectedOmega[0,1:(NUMSTATES+1)] = np.array(InitOmega)

    expectedXBar = np.zeros([2001,NUMSTATES+1])
    expectedXBar[0,1:(NUMSTATES+1)] = np.array(Initialx)

    expectedSTM = np.zeros([2001,NUMSTATES,NUMSTATES])
    expectedSTM[0,:,:] = np.eye(NUMSTATES)

    expectedCovar = np.zeros([2001,NUMSTATES*NUMSTATES+1])
    expectedCovar[0,1:(NUMSTATES*NUMSTATES+1)] = np.array(InitialCovar)

    expDynMat = np.zeros([2001,NUMSTATES,NUMSTATES])
    Gamma = dt ** 2. / 2. * np.eye(3)
    ProcNoiseCovar = np.dot(Gamma, np.dot(module.qProcVal*np.eye(3),Gamma.T))

    for i in range(1,2001):
        expectedStateArray[i,0] = dt*i*1E9
        expectedPrevArray[i,0] = dt*i*1E9
        expectedOmega[i,0] = dt*i*1E9
        expectedCovar[i,0] =  dt*i*1E9
        expectedXBar[i,0] = dt*i*1E9

        #Simulate sunline Dyn Mat
        expDynMat[i-1, :, :] = - np.array([[0., -expectedOmega[i-1, 3], expectedOmega[i-1,2]],
                                         [expectedOmega[i-1,3], 0., -expectedOmega[i-1,1]],
                                         [ -expectedOmega[i-1,2], expectedOmega[i-1,1], 0.]])

        #Simulate STM State prop
        expectedStateArray[i,1:(NUMSTATES+1)] =  np.array(expectedStateArray[i-1,1:(NUMSTATES+1)] - dt*(np.cross(np.array(expectedOmega[i-1,1:(NUMSTATES+1)]), np.array(expectedStateArray[i-1,1:(NUMSTATES+1)]))))
        expectedPrevArray[i, 1:(NUMSTATES + 1)] = expectedStateArray[i-1,1:(NUMSTATES+1)]
        expectedSTM[i,:,:] = dt * np.dot(expDynMat[i-1,:,:], np.eye(NUMSTATES)) + np.eye(NUMSTATES)

        # Simulate Rate compute
        normdk = np.linalg.norm(expectedStateArray[i, 1:(NUMSTATES + 1)])
        nomrdkmin1 = np.linalg.norm(expectedPrevArray[i, 1:(NUMSTATES + 1)])
        arg = np.dot(expectedStateArray[i, 1:(NUMSTATES + 1)], expectedPrevArray[i , 1:(NUMSTATES + 1)]) / (normdk * nomrdkmin1)
        if arg>1:
            expectedOmega[i, 1:(NUMSTATES + 1)] = 1./dt*np.cross(expectedStateArray[i, 1:(NUMSTATES + 1)],
                                                           expectedPrevArray[i, 1:(NUMSTATES + 1)]) / (normdk * nomrdkmin1) * np.arccos(1)
        elif arg<-1:
            expectedOmega[i, 1:(NUMSTATES + 1)] = 1./dt*np.cross(expectedStateArray[i, 1:(NUMSTATES + 1)],
                                                           expectedPrevArray[i, 1:(NUMSTATES + 1)]) / (
                                                  normdk * nomrdkmin1) * np.arccos(-1)

        else:
            expectedOmega[i, 1:(NUMSTATES + 1)] = 1./dt*np.cross(expectedStateArray[i, 1:(NUMSTATES + 1)],expectedPrevArray[i, 1:(NUMSTATES + 1)]) / (normdk * nomrdkmin1) * np.arccos(arg)

        expectedXBar[i, 1:(NUMSTATES+1)] = np.dot(expectedSTM[i, :, :], expectedXBar[i - 1, 1:(NUMSTATES+1)])
        expectedCovar[i,1:(NUMSTATES*NUMSTATES+1)] = (np.dot(expectedSTM[i,:,:], np.dot(np.reshape(expectedCovar[i-1,1:(NUMSTATES*NUMSTATES+1)],[NUMSTATES,NUMSTATES]), np.transpose(expectedSTM[i,:,:])))+ ProcNoiseCovar).flatten()

    FilterPlots.StatesVsExpected(stateLog, expectedStateArray, show_plots)
    FilterPlots.StatesPlotCompare(stateErrorLog, expectedXBar, covarLog, expectedCovar, show_plots)
    FilterPlots.OmegaVsExpected(expectedOmega, omegaLog, show_plots)

    for j in range(1,2001):
        for i in range(NUMSTATES):
            if (abs(stateLog[j, i + 1] - expectedStateArray[j, i + 1]) > 1.0E-10):
                testFailCount += 1
                testMessages.append("General state propagation failure: State Prop \n")
            if (abs(stateErrorLog[j, i + 1] - expectedXBar[j, i + 1]) > 1.0E-10):
                testFailCount += 1
                testMessages.append("General state propagation failure: State Error Prop \n")

        for i in range(NUMSTATES*NUMSTATES):
            if (abs(covarLog[j, i + 1] - expectedCovar[j, i + 1]) > 1.0E-8):
                print(abs(covarLog[j, i + 1] - expectedCovar[j, i + 1]))
                abs(covarLog[j, i + 1] - expectedCovar[j, i + 1])
                testFailCount += 1
                # testMessages.append("General state propagation failure: Covariance Prop \n")
            if (abs(stmLog[j, i + 1] - expectedSTM[j,:].flatten()[i]) > 1.0E-10):
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
    NUMSTATES=3

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
    module = okeefeEKF.okeefeEKF()
    module.ModelTag = "okeefeEKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)
    setupFilterData(module)
    module.omega = [0.,0.,0.]

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
    CSSBias = [1.0 for i in range(len(CSSOrientationList))]
    totalCSSList = []
    i=0
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

    dt =0.5
    stateTarget1 = testVector1
    module.state = stateGuess
    module.x = (np.array(stateTarget1) - np.array(stateGuess)).tolist()
    kfLog = module.logger("x", testProcessRate)
    unitTestSim.AddModelToTask(unitTaskName, kfLog)
    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

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

    stateLog = dataLog.state
    covarLog = dataLog.covar

    if not AddMeasNoise:
        for i in range(NUMSTATES):
            if (abs(covarLog[-1, i * NUMSTATES + i] - covarLog[0, i * NUMSTATES  + i] / 100.) > 1E-1):
                testFailCount += 1
                testMessages.append("Covariance update failure")
            if (abs(stateLog[-1, i] - stateTarget1[i]) > 1.0E-10):
                testFailCount += 1
                testMessages.append("State update failure")
    else:
        for i in range(NUMSTATES):
            if (abs(covarLog[-1, i * NUMSTATES + i] - covarLog[0, i * NUMSTATES + i] / 100.) > 1E-1):
                testFailCount += 1
                testMessages.append("Covariance update failure with noise")
            if (abs(stateLog[-1, i] - stateTarget1[i]) > 1.0E-2):
                testFailCount += 1
                testMessages.append("State update failure with noise")


    stateTarget2 = testVector2

    inputData = messaging.CSSArraySensorMsgPayload()
    for i in range(SimHalfLength):
        if i > 20:
            dotList = []
            for element in CSSOrientationList:
                if AddMeasNoise:
                    dotProd = np.dot(np.array(element), np.array(testVector2)[0:3]) + np.random.normal(0., module.qObsVal)
                else:
                    dotProd = np.dot(np.array(element), np.array(testVector2)[0:3])
                dotList.append(dotProd)
            inputData.CosValue = dotList
            cssDataInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)

        unitTestSim.ConfigureStopTime(macros.sec2nano((i + SimHalfLength+1) * 0.5))
        unitTestSim.ExecuteSimulation()

    stateErrorLog = unitTestSupport.addTimeColumn(kfLog.times(), kfLog.x)
    stateLog = dataLog.state
    postFitLog = dataLog.postFitRes
    covarLog = dataLog.covar


    if not AddMeasNoise:
        for i in range(NUMSTATES):
            if (abs(covarLog[-1, i * NUMSTATES + i] - covarLog[0, i * NUMSTATES + i] / 100.) > 1E-1):
                testFailCount += 1
                testMessages.append("Covariance update failure")
            if (abs(stateLog[-1, i] - stateTarget2[i]) > 1.0E-10):
                testFailCount += 1
                testMessages.append("State update failure")
    else:
        for i in range(NUMSTATES):
            if (abs(covarLog[-1, i * NUMSTATES + i] - covarLog[0, i * NUMSTATES + i] / 100.) > 1E-1):
                testFailCount += 1
                testMessages.append("Covariance update failure")
            if (abs(stateLog[-1, i] - stateTarget2[i]) > 1.0E-2):
                testFailCount += 1
                testMessages.append("State update failure")


    target1 = np.array(testVector1)
    target2 = np.array(testVector2)
    FilterPlots.StatesPlot(dataLog.times(), stateErrorLog, covarLog, show_plots)
    FilterPlots.StatesVsTargets(dataLog.times(), target1, target2, stateLog, show_plots)
    FilterPlots.PostFitResiduals(dataLog.times(), postFitLog, module.qObsVal, show_plots)

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + "EKF full test")
    else:
        print((testMessages))

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]



if __name__ == "__main__":
    StateUpdateSunLine(True, 200, True ,[-0.7, 0.7, 0.0] ,[0.8, 0.9, 0.0], [0.7, 0.7, 0.0])
    # sunline_individual_test()
    # StatePropStatic()
    # StatePropVariable(True)
