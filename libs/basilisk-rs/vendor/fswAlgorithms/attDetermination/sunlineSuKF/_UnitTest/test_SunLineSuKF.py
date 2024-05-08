''' '''
'''
 ISC License

 Copyright (c) 2016-2018, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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
import numpy
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import sunlineSuKF  # import the module that is to be tested
from Basilisk.utilities import SimulationBaseClass, macros

import SunLineSuKF_test_utilities as FilterPlots


def addTimeColumn(time, data):
    return numpy.transpose(numpy.vstack([[time], numpy.transpose(data)]))

def setupFilterData(filterObject, initialized):
    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0

    if initialized:
        filterObject.stateInit = [0.0, 0.0, 1.0, 0.0, 0.0, 1.]
        filterObject.filterInitialized = 1
    else:
        filterObject.filterInitialized = 0

    filterObject.covarInit = [1., 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 1., 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 1., 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.02, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.02, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 1E-4]
    qNoiseIn = numpy.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3]*0.001*0.001
    qNoiseIn[3:5, 3:5] = qNoiseIn[3:5, 3:5]*0.001*0.001
    qNoiseIn[5, 5] = qNoiseIn[5, 5]*0.0000002*0.0000002
    filterObject.qNoise = qNoiseIn.reshape(36).tolist()
    filterObject.qObsVal = 0.002
    filterObject.sensorUseThresh = 0.0


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
@pytest.mark.parametrize("kellyOn", [
    (False),
    (True)
])


def test_all_sunline_kf(show_plots, kellyOn):
    """Module Unit Test"""
    [testResults, testMessage] = SwitchMethods()
    assert testResults < 1, testMessage
    [testResults, testMessage] = StatePropSunLine(show_plots)
    assert testResults < 1, testMessage
    [testResults, testMessage] = StateUpdateSunLine(show_plots, kellyOn)
    assert testResults < 1, testMessage
    [testResults, testMessage] = FaultScenarios()
    assert testResults < 1, testMessage


def SwitchMethods():
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages
    ###################################################################################
    ## Test the sunlineSEKFComputeDCM_BS method
    ###################################################################################
    numStates = 6

    inputStates = [2, 1, 0.75, 0.1, 0.4, 0.]
    sunheading = inputStates[:3]
    bvec1 = [0., 1., 0.]
    b1 = numpy.array(bvec1)

    dcm_BS = [1., 0., 0.,
             0., 1., 0.,
             0., 0., 1.]

    # Fill in expected values for test

    DCM_exp = numpy.zeros([3,3])
    W_exp = numpy.eye(numStates)

    DCM_exp[:, 0] = numpy.array(inputStates[0:3]) / (numpy.linalg.norm(numpy.array(inputStates[0:3])))
    DCM_exp[:, 1] = numpy.cross(DCM_exp[:, 0], b1) / numpy.linalg.norm(numpy.array(numpy.cross(DCM_exp[:, 0], b1)))
    DCM_exp[:, 2] = numpy.cross(DCM_exp[:, 0], DCM_exp[:, 1]) / numpy.linalg.norm(
        numpy.cross(DCM_exp[:, 0], DCM_exp[:, 1]))

    # Fill in the variables for the test
    dcm = sunlineSuKF.new_doubleArray(3 * 3)

    for j in range(9):
        sunlineSuKF.doubleArray_setitem(dcm, j, dcm_BS[j])

    sunlineSuKF.sunlineSuKFComputeDCM_BS(sunheading, bvec1, dcm)

    switchBSout = []
    dcmOut = []
    for j in range(9):
        dcmOut.append(sunlineSuKF.doubleArray_getitem(dcm, j))


    errorNorm = numpy.zeros(1)
    errorNorm[0] = numpy.linalg.norm(DCM_exp - numpy.array(dcmOut).reshape([3, 3]))

    for i in range(len(errorNorm)):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("Frame switch failure \n")

    ###################################################################################
    ## Test the Switching method
    ###################################################################################

    inputStates = [2,1,0.75,0.1,0.4, 1.]
    bvec1 = [0.,1.,0.]
    b1 = numpy.array(bvec1)
    covar = [1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.,
             1., 0., 0., 1., 0., 0.,
             0., 1., 0., 0., 1., 0.,
             0., 0., 1., 0., 0., 1.]
    noise =0.01

    # Fill in expected values for test

    DCM_BSold = numpy.zeros([3,3])
    DCM_BSnew = numpy.zeros([3,3])
    Switch = numpy.eye(numStates)
    SwitchBSold = numpy.eye(numStates)
    SwitchBSnew = numpy.eye(numStates)

    DCM_BSold[:,0] = numpy.array(inputStates[0:3])/(numpy.linalg.norm(numpy.array(inputStates[0:3])))
    DCM_BSold[:,1] = numpy.cross(DCM_BSold[:,0], b1)/numpy.linalg.norm(numpy.array(numpy.cross(DCM_BSold[:,0], b1)))
    DCM_BSold[:,2] = numpy.cross(DCM_BSold[:,0], DCM_BSold[:,1])/numpy.linalg.norm(numpy.cross(DCM_BSold[:,0], DCM_BSold[:,1]))
    SwitchBSold[3:5, 3:5] = DCM_BSold[1:3, 1:3]

    b2 = numpy.array([1.,0.,0.])
    DCM_BSnew[:,0] = numpy.array(inputStates[0:3])/(numpy.linalg.norm(numpy.array(inputStates[0:3])))
    DCM_BSnew[:,1] = numpy.cross(DCM_BSnew[:,0], b2)/numpy.linalg.norm(numpy.array(numpy.cross(DCM_BSnew[:,0], b2)))
    DCM_BSnew[:,2] = numpy.cross(DCM_BSnew[:,0], DCM_BSnew[:,1])/numpy.linalg.norm(numpy.cross(DCM_BSnew[:,0], DCM_BSnew[:,1]))
    SwitchBSnew[3:5, 3:5] = DCM_BSnew[1:3, 1:3]

    DCM_newOld = numpy.dot(DCM_BSnew.T, DCM_BSold)
    Switch[3:5, 3:5] = DCM_newOld[1:3,1:3]

    # Fill in the variables for the test
    bvec = sunlineSuKF.new_doubleArray(3)
    states = sunlineSuKF.new_doubleArray(numStates)
    covarMat = sunlineSuKF.new_doubleArray(numStates * numStates)

    for i in range(3):
        sunlineSuKF.doubleArray_setitem(bvec, i, bvec1[i])
    for i in range(numStates):
        sunlineSuKF.doubleArray_setitem(states, i, inputStates[i])
    for j in range(numStates*numStates):
        sunlineSuKF.doubleArray_setitem(covarMat, j, covar[j])
        # sunlineSEKF.doubleArray_setitem(switchBS, j, switchInput[j])

    sunlineSuKF.sunlineSuKFSwitch(bvec, states, covarMat)

    switchBSout = []
    covarOut = []
    stateOut = []
    bvecOut = []
    for i in range(3):
        bvecOut.append(sunlineSuKF.doubleArray_getitem(bvec, i))
    for i in range(numStates):
        stateOut.append(sunlineSuKF.doubleArray_getitem(states, i))
    for j in range(numStates*numStates):
        covarOut.append(sunlineSuKF.doubleArray_getitem(covarMat, j))


    expectedState = numpy.dot(Switch, numpy.array(inputStates))
    Pk = numpy.array(covar).reshape([numStates, numStates])
    expectedP = numpy.dot(Switch, numpy.dot(Pk, Switch.T))

    errorNorm = numpy.zeros(3)
    errorNorm[0] = numpy.linalg.norm(numpy.array(stateOut) - expectedState)
    errorNorm[1] = numpy.linalg.norm(expectedP - numpy.array(covarOut).reshape([numStates, numStates]))
    errorNorm[2] = numpy.linalg.norm(numpy.array(bvecOut) - b2)

    for i in range(len(errorNorm)):
        if (errorNorm[i] > 1.0E-10):
            testFailCount += 1
            testMessages.append("Frame switch failure \n")


    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + " SuKF switch tests")
    else:
        print(str(testFailCount) + ' tests failed')
        print(testMessages)
    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def StateUpdateSunLine(show_plots, kellyOn):
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
    module = sunlineSuKF.sunlineSuKF()
    module.ModelTag = "sunlineSuKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module, False)
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
    totalCSSList = []
    for CSSHat in CSSOrientationList:
        newCSS = messaging.CSSUnitConfigMsgPayload()
        newCSS.CBias = 1.0
        newCSS.nHat_B = CSSHat
        totalCSSList.append(newCSS)
    cssConstelation.nCSS = len(CSSOrientationList)
    cssConstelation.cssVals = totalCSSList
    cssConstInMsg = messaging.CSSConfigMsg().write(cssConstelation)

    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Add the kelly curve coefficients
    if kellyOn:
        kellList = []
        for j in range(len(CSSOrientationList)):
            kellyData = sunlineSuKF.SunlineSuKFCFit()
            kellyData.cssKellFact = 0.05
            kellyData.cssKellPow = 2.
            kellyData.cssRelScale = 1.
            kellList.append(kellyData)
        module.kellFits = kellList

    testVector = numpy.array([-0.7, 0.7, 0.0])
    testVector/=numpy.linalg.norm(testVector)
    inputData = messaging.CSSArraySensorMsgPayload()
    dotList = []
    for element in CSSOrientationList:
        dotProd = numpy.dot(numpy.array(element), testVector)/(numpy.linalg.norm(element)*numpy.linalg.norm(testVector))
        dotList.append(dotProd)

    inputData.CosValue = dotList
    cssDataInMsg = messaging.CSSArraySensorMsg()

    stateTarget = testVector.tolist()
    stateTarget.extend([0.0, 0.0, 1.])
    # module.stateInit = [0.7, 0.7, 0.0, 0.01, 0.001, 1.]

    # connect messages
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConstInMsg)

    numStates = len(module.stateInit)
    unitTestSim.InitializeSimulation()
    if kellyOn:
        time = 1000
    else:
        time =  500
    for i in range(time):
        cssDataInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i+1)*0.5))
        unitTestSim.ExecuteSimulation()

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    postFitLog = addTimeColumn(dataLog.times(), dataLog.postFitRes)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    accuracy = 1.0E-3
    if kellyOn:
        accuracy = 1.0E-2 # 1% Error test for the kelly curves given errors
    for i in range(numStates):
        if(covarLog[-1, i*numStates+1+i] > covarLog[0, i*numStates+1+i]):
            testFailCount += 1
            testMessages.append("Covariance update failure first part")
    if(numpy.arccos(numpy.dot(stateLog[-1, 1:4], stateTarget[0:3])/(numpy.linalg.norm(stateLog[-1, 1:4])*numpy.linalg.norm(stateTarget[0:3]))) > accuracy):
        print(numpy.arccos(numpy.dot(stateLog[-1, 1:4], stateTarget[0:3])/(numpy.linalg.norm(stateLog[-1, 1:4])*numpy.linalg.norm(stateTarget[0:3]))))
        testFailCount += 1
        testMessages.append("Pointing update failure")
    if(numpy.linalg.norm(stateLog[-1, 4:7] - stateTarget[3:6]) > accuracy):
        print(numpy.linalg.norm(stateLog[-1,  4:7] - stateTarget[3:6]))
        testFailCount += 1
        testMessages.append("Rate update failure")
    if(abs(stateLog[-1, 6] - stateTarget[5]) > accuracy):
        print(abs(stateLog[-1, 6] - stateTarget[5]))
        testFailCount += 1
        testMessages.append("Sun Intensity update failure")

    testVector = numpy.array([-0.7, 0.75, 0.0])
    testVector /= numpy.linalg.norm(testVector)
    inputData = messaging.CSSArraySensorMsgPayload()
    dotList = []
    for element in CSSOrientationList:
        dotProd = numpy.dot(numpy.array(element), testVector)
        dotList.append(dotProd)
    inputData.CosValue = dotList

    for i in range(time):
        if i > 20:
            cssDataInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i+time+1)*0.5))
        unitTestSim.ExecuteSimulation()

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    postFitLog = addTimeColumn(dataLog.times(), dataLog.postFitRes)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    stateTarget = testVector.tolist()
    stateTarget.extend([0.0, 0.0, 1.0])

    for i in range(numStates):
        if(covarLog[-1, i*numStates+1+i] > covarLog[0, i*numStates+1+i]):
            print(covarLog[-1, i*numStates+1+i] - covarLog[0, i*numStates+1+i])
            testFailCount += 1
            testMessages.append("Covariance update failure")
    if(numpy.arccos(numpy.dot(stateLog[-1, 1:4], stateTarget[0:3])/(numpy.linalg.norm(stateLog[-1, 1:4])*numpy.linalg.norm(stateTarget[0:3]))) > accuracy):
        print(numpy.arccos(numpy.dot(stateLog[-1, 1:4], stateTarget[0:3])/(numpy.linalg.norm(stateLog[-1, 1:4])*numpy.linalg.norm(stateTarget[0:3]))))
        testFailCount += 1
        testMessages.append("Pointing update failure")
    if(numpy.linalg.norm(stateLog[-1, 4:7] - stateTarget[3:6]) > accuracy):
        print(numpy.linalg.norm(stateLog[-1,  4:7] - stateTarget[3:6]))
        testFailCount += 1
        testMessages.append("Rate update failure")
    if(abs(stateLog[-1, 6] - stateTarget[5]) > accuracy):
        print(abs(stateLog[-1, 6] - stateTarget[5]))
        testFailCount += 1
        testMessages.append("Sun Intensity update failure")

    FilterPlots.StateCovarPlot(stateLog, covarLog, show_plots)
    FilterPlots.PostFitResiduals(postFitLog, module.qObsVal, show_plots)

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state update")
    else:
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def StatePropSunLine(show_plots):
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
    module = sunlineSuKF.sunlineSuKF()
    module.ModelTag = "sunlineSuKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module, True)
    numStates = 6
    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    cssConstInMsg = messaging.CSSConfigMsg()
    cssDataInMsg = messaging.CSSArraySensorMsg()

    # connect messages
    module.cssDataInMsg.subscribeTo(cssDataInMsg)
    module.cssConfigInMsg.subscribeTo(cssConstInMsg)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(8000.0))
    unitTestSim.ExecuteSimulation()

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    postFitLog = addTimeColumn(dataLog.times(), dataLog.postFitRes)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    FilterPlots.StateCovarPlot(stateLog, covarLog, show_plots)
    FilterPlots.PostFitResiduals(postFitLog, module.qObsVal, show_plots)

    for i in range(numStates):
        if(abs(stateLog[-1, i+1] - stateLog[0, i+1]) > 1.0E-10):
            print(abs(stateLog[-1, i+1] - stateLog[0, i+1]))
            testFailCount += 1
            testMessages.append("State propagation failure")



    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state propagation")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def FaultScenarios():
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

    # Clean methods for Measurement and Time Updates
    moduleClean1 = sunlineSuKF.SunlineSuKFConfig()
    moduleClean1.numStates = 6
    moduleClean1.countHalfSPs = moduleClean1.numStates
    moduleClean1.state = [0., 0., 0., 0., 0., 0.]
    moduleClean1.statePrev = [0., 0., 0., 0., 0., 0.]
    moduleClean1.sBar = [0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.]
    moduleClean1.sBarPrev = [1., 0., 0., 0., 0., 0.,
                                   0., 1., 0., 0., 0., 0.,
                                   0., 0., 1., 0., 0., 0.,
                                   0., 0., 0., 1., 0., 0.,
                                   0., 0., 0., 0., 1., 0.,
                                   0., 0., 0., 0., 0., 1.]
    moduleClean1.covar = [0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.]
    moduleClean1.covarPrev = [2., 0., 0., 0., 0., 0.,
                                    0., 2., 0., 0., 0., 0.,
                                    0., 0., 2., 0., 0., 0.,
                                    0., 0., 0., 2., 0., 0.,
                                    0., 0., 0., 0., 2., 0.,
                                    0., 0., 0., 0., 0., 2.]

    sunlineSuKF.sunlineSuKFCleanUpdate(moduleClean1)

    if numpy.linalg.norm(numpy.array(moduleClean1.covarPrev) - numpy.array(moduleClean1.covar)) > 1E10:
        testFailCount += 1
        testMessages.append("sunlineSuKFClean Covar failed")
    if numpy.linalg.norm(numpy.array(moduleClean1.statePrev) - numpy.array(moduleClean1.state)) > 1E10:
        testFailCount += 1
        testMessages.append("sunlineSuKFClean States failed")
    if numpy.linalg.norm(numpy.array(moduleClean1.sBar) - numpy.array(moduleClean1.sBarPrev)) > 1E10:
        testFailCount += 1
        testMessages.append("sunlineSuKFClean sBar failed")

    cssConstInMsg = messaging.CSSConfigMsg()
    cssDataInMsg = messaging.CSSArraySensorMsg()

    # connect messages
    moduleClean1.cssDataInMsg.subscribeTo(cssDataInMsg)
    moduleClean1.cssConfigInMsg.subscribeTo(cssConstInMsg)

    moduleClean1.alpha = 0.02
    moduleClean1.beta = 2.0
    moduleClean1.kappa = 0.0

    moduleClean1.wC = [-1] * (moduleClean1.numStates * 2 + 1)
    moduleClean1.wM = [-1] * (moduleClean1.numStates * 2 + 1)
    retTime = sunlineSuKF.sunlineSuKFTimeUpdate(moduleClean1, 1)
    retMease = sunlineSuKF.sunlineSuKFMeasUpdate(moduleClean1, 1)
    if retTime == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Time update")
    if retMease == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Meas update")


    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: fault detection test")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

if __name__ == "__main__":
    # test_all_sunline_kf(True)
    # StateUpdateSunLine(True, True)
    FaultScenarios()
