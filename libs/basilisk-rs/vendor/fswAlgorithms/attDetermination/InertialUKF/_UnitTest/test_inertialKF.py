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
import math
import os

import matplotlib.pyplot as plt
import numpy
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import inertialUKF  # import the module that is to be tested
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
textSnippetPassed = r'\textcolor{ForestGreen}{' + "PASSED" + '}'
textSnippetFailed = r'\textcolor{Red}{' + "Failed" + '}'

def setupFilterData(filterObject):

    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0
    filterObject.switchMag = 1.2

    ST1Data = inertialUKF.STMessage()

    ST1Data.noise = [0.00017 * 0.00017, 0.0, 0.0,
                     0.0, 0.00017 * 0.00017, 0.0,
                     0.0, 0.0, 0.00017 * 0.00017]

    ST2Data = inertialUKF.STMessage()

    ST2Data.noise = [0.00017 * 0.00017, 0.0, 0.0,
                     0.0, 0.00017 * 0.00017, 0.0,
                     0.0, 0.0, 0.00017 * 0.00017]
    STList = [ST1Data, ST2Data]
    filterObject.STDatasStruct.STMessages = STList
    filterObject.STDatasStruct.numST = len(STList)

    filterObject.stateInit = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    filterObject.covarInit = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.04, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.04, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.004, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.004, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.004]
    qNoiseIn = numpy.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3]*0.0017*0.0017
    qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6]*0.00017*0.00017
    filterObject.qNoise = qNoiseIn.reshape(36).tolist()

    lpDataUse = inertialUKF.LowPassFilterData()
    lpDataUse.hStep = 0.5
    lpDataUse.omegCutoff = 15.0/(2.0*math.pi)
    filterObject.gyroFilt = [lpDataUse, lpDataUse, lpDataUse]

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def all_inertial_kfTest(show_plots):
    """Module Unit Tests"""
    # the following two tests appear to be broken
    # [testResults, testMessage] = statePropInertialAttitude(show_plots)
    # assert testResults < 1, testMessage
    # [testResults, testMessage] = statePropRateInertialAttitude(show_plots)
    # assert testResults < 1, testMessage
    [testResults, testMessage] = stateUpdateInertialAttitude(show_plots)
    assert testResults < 1, testMessage
    [testResults, testMessage] = stateUpdateRWInertialAttitude(show_plots)
    assert testResults < 1, testMessage
    [testResults, testMessage] = filterMethods()
    assert testResults < 1, testMessage
    # [testResults, testMessage] = faultScenarios()
    # assert testResults < 1, testMessage

def test_FilterMethods():
    [testResults, testMessage] = filterMethods()
    assert testResults < 1, testMessage
def filterMethods():
    """Module Unit Test"""
    testFailCount = 0
    testMessages = []

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(1.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    accuracy = 1E-10
    # Construct algorithm and associated C++ container
    module = inertialUKF.inertialUKF()
    module.ModelTag = "inertialUKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    st1 = messaging.STAttMsgPayload()
    st1.timeTag = macros.sec2nano(1.25)
    st1.MRP_BdyInrtl = [0.1, 0.2, 0.3]
    st2 = messaging.STAttMsgPayload()
    st2.timeTag = macros.sec2nano(1.0)
    st2.MRP_BdyInrtl = [0.2, 0.2, 0.3]
    st3 = messaging.STAttMsgPayload()
    st3.timeTag = macros.sec2nano(0.75)
    st3.MRP_BdyInrtl = [0.3, 0.2, 0.3]

    ST1Data = inertialUKF.STMessage()
    ST2Data = inertialUKF.STMessage()
    ST3Data = inertialUKF.STMessage()

    STList = [ST1Data, ST2Data, ST3Data]

    state = inertialUKF.new_doubleArray(6)
    stateInput = numpy.array([1., 0., 0., 0.1, 0.1, 0.1])
    for i in range(len(stateInput)):
        inertialUKF.doubleArray_setitem(state, i, stateInput[i])

    wheelAccel = numpy.array([-5, 5]) / 1. * numpy.array([1., 1])
    angAccel = -0.5 * (wheelAccel[0] + wheelAccel[1]) * numpy.array([1., 0., 0])
    expectedRate = numpy.array(stateInput[3:]) + angAccel

    inertialUKF.inertialStateProp(module.getConfig(), state, 0.5)
    stateOut = []
    for j in range(6):
        stateOut.append(inertialUKF.doubleArray_getitem(state, j))

    if numpy.linalg.norm(expectedRate - numpy.array(stateOut)[3:]) > accuracy:
        testFailCount += 1
        testMessages.append("Failed to capture wheel acceleration in inertialStateProp")

    setupFilterData(module)
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
     0., 800., 0.,
     0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    module.STDatasStruct.STMessages = STList
    module.STDatasStruct.numST = len(STList)

    inertialUKFLog = module.logger("stSensorOrder")
    unitTestSim.AddModelToTask(unitTaskName, inertialUKFLog)

    # create ST input messages
    st1InMsg = messaging.STAttMsg().write(st1)
    st2InMsg = messaging.STAttMsg().write(st2)
    st3InMsg = messaging.STAttMsg().write(st3)

    # make input messages but don't write to them
    rwSpeedInMsg = messaging.RWSpeedMsg()
    rwConfigInMsg = messaging.RWArrayConfigMsg()
    gyroInMsg = messaging.AccDataMsg()

    # connect messages
    module.STDatasStruct.STMessages[0].stInMsg.subscribeTo(st1InMsg)
    module.STDatasStruct.STMessages[1].stInMsg.subscribeTo(st2InMsg)
    module.STDatasStruct.STMessages[2].stInMsg.subscribeTo(st3InMsg)
    module.massPropsInMsg.subscribeTo(vcInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.gyrBuffInMsg.subscribeTo(gyroInMsg)

    # Star Tracker Read Message and Order method
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(1E9)
    unitTestSim.ExecuteSimulation()

    stOrdered = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.stSensorOrder)
    if numpy.linalg.norm(numpy.array(stOrdered[0]) - numpy.array([0., 2, 1, 0, 0])) > accuracy:
        testFailCount += 1
        testMessages.append("ST order test failed")

    unitTestSupport.writeTeXSnippet("toleranceValue00", str(accuracy), path)
    if testFailCount == 0:
        print('Passed: test_FilterMethods')
        unitTestSupport.writeTeXSnippet("passFail00", textSnippetPassed, path)
    else:
        print('Failed: test_FilterMethods')
        unitTestSupport.writeTeXSnippet("passFail00", textSnippetFailed, path)

    return [testFailCount, ''.join(testMessages)]

def test_stateUpdateInertialAttitude(show_plots):
    [testResults, testMessage] = stateUpdateInertialAttitude(show_plots)
    assert testResults < 1, testMessage
def stateUpdateInertialAttitude(show_plots):
    """Module Unit Test"""
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
    module = inertialUKF.inertialUKF()
    module.ModelTag = "InertialUKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)
    module.maxTimeJump = 10

    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    stMessage1 = messaging.STAttMsgPayload()
    stMessage1.MRP_BdyInrtl = [0.3, 0.4, 0.5]
    st1InMsg = messaging.STAttMsg()

    stMessage2 = messaging.STAttMsgPayload()
    stMessage2.MRP_BdyInrtl = [0.3, 0.4, 0.5]
    st2InMsg = messaging.STAttMsg()

#    stateTarget = testVector.tolist()
#    stateTarget.extend([0.0, 0.0, 0.0])
#    module.state = [0.7, 0.7, 0.0]
    inertialUKFLog = module.logger(["covar", "state"], testProcessRate*10)
    unitTestSim.AddModelToTask(unitTaskName, inertialUKFLog)

    # make input messages but don't write to them
    rwSpeedInMsg = messaging.RWSpeedMsg()
    rwConfigInMsg = messaging.RWArrayConfigMsg()
    gyroInMsg = messaging.AccDataMsg()

    # connect messages
    module.STDatasStruct.STMessages[0].stInMsg.subscribeTo(st1InMsg)
    module.STDatasStruct.STMessages[1].stInMsg.subscribeTo(st2InMsg)
    module.massPropsInMsg.subscribeTo(vcInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.gyrBuffInMsg.subscribeTo(gyroInMsg)

    unitTestSim.InitializeSimulation()

    for i in range(20000):
        if i > 21:
            stMessage1.timeTag = int(i*0.5*1E9)
            stMessage2.timeTag = int(i*0.5*1E9)
            st1InMsg.write(stMessage1, unitTestSim.TotalSim.CurrentNanos)
            st2InMsg.write(stMessage2, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i+1)*0.5))
        unitTestSim.ExecuteSimulation()

    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    stateLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.state)
    accuracy = 1.0E-5
    unitTestSupport.writeTeXSnippet("toleranceValue11", str(accuracy), path)
    for i in range(3):
        if(covarLog[-1, i*6+1+i] > covarLog[0, i*6+1+i]):
            testFailCount += 1
            testMessages.append("Covariance update failure")
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetPassed, path)
        if(abs(stateLog[-1, i+1] - stMessage1.MRP_BdyInrtl[i]) > accuracy):
            print(abs(stateLog[-1, i+1] - stMessage1.MRP_BdyInrtl[i]))
            testFailCount += 1
            testMessages.append("State update failure")
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetPassed, path)

    stMessage1.MRP_BdyInrtl = [1.2, 0.0, 0.0]
    stMessage2.MRP_BdyInrtl = [1.2, 0.0, 0.0]

    for i in range(20000):
        if i > 20:
            stMessage1.timeTag = int((i+20000)*0.25*1E9)
            stMessage2.timeTag = int((i+20000)*0.5*1E9)
            st1InMsg.write(stMessage1, unitTestSim.TotalSim.CurrentNanos)
            st2InMsg.write(stMessage2, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i+20000+1)*0.5))
        unitTestSim.ExecuteSimulation()


    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    stateLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.state)
    for i in range(3):
        if(covarLog[-1, i*6+1+i] > covarLog[0, i*6+1+i]):
            testFailCount += 1
            testMessages.append("Covariance update large failure")
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetPassed, path)
    plt.figure()
    for i in range(module.numStates):
        plt.plot(stateLog[:,0]*1.0E-9, stateLog[:,i+1], label='State_' +str(i))
        plt.legend()
        plt.ylim([-1, 1])

    unitTestSupport.writeFigureLaTeX('Test11', 'Test 1 State convergence', plt, 'width=0.9\\textwidth, keepaspectratio', path)
    plt.figure()
    for i in range(module.numStates):
        plt.plot(covarLog[:,0]*1.0E-9, covarLog[:,i*module.numStates+i+1], label='Covar_' +str(i))
        plt.legend()
        plt.ylim([0, 2.E-7])

    unitTestSupport.writeFigureLaTeX('Test12', 'Test 1 Covariance convergence', plt, 'width=0.9\\textwidth, keepaspectratio', path)
    if(show_plots):
        plt.show()
        plt.close('all')

    # print out success message if no error were found
    if testFailCount == 0:
        print('Passed: test_StateUpdateInertialAttitude')
    else:
        print('Failed: test_StateUpdateInertialAttitude')

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def BROKENtest_statePropInertialAttitude(show_plots):
    [testResults, testMessage] = statePropInertialAttitude(show_plots)
    assert testResults < 1, testMessage
def statePropInertialAttitude(show_plots):
    """Module Unit Test"""

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
    module = inertialUKF.inertialUKF()
    module.ModelTag = "InertialUKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)


    inertialUKFLog = module.logger(["covar", "state"], testProcessRate*10)

    # make input messages but don't write to them
    rwSpeedInMsg = messaging.RWSpeedMsg()
    rwConfigInMsg = messaging.RWArrayConfigMsg()
    gyroInMsg = messaging.AccDataMsg()
    st1InMsg = messaging.STAttMsg()
    st2InMsg = messaging.STAttMsg()

    # connect messages
    module.STDatasStruct.STMessages[0].stInMsg.subscribeTo(st1InMsg)
    module.STDatasStruct.STMessages[1].stInMsg.subscribeTo(st2InMsg)
    module.massPropsInMsg.subscribeTo(vcInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.gyrBuffInMsg.subscribeTo(gyroInMsg)


    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(8000.0))
    unitTestSim.ExecuteSimulation()

    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    stateLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.state)

    accuracy = 1.0E-10
    unitTestSupport.writeTeXSnippet("toleranceValue22", str(accuracy), path)
    for i in range(6):
        if(abs(stateLog[-1, i+1] - stateLog[0, i+1]) > accuracy):
            testFailCount += 1
            testMessages.append("State propagation failure")
            unitTestSupport.writeTeXSnippet('passFail22', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail22', textSnippetPassed, path)

    for i in range(6):
       if(covarLog[-1, i*6+i+1] <= covarLog[0, i*6+i+1]):
           testFailCount += 1
           testMessages.append("State covariance failure i="+str(i))

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state propagation")
    else:
        print('Failed: test_StatePropInertialAttitude')
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def test_stateUpdateRWInertialAttitude(show_plots):
    [testResults, testMessage] = stateUpdateRWInertialAttitude(show_plots)
    assert testResults < 1, testMessage
def stateUpdateRWInertialAttitude(show_plots):
    """Module Unit Test"""
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
    module = inertialUKF.inertialUKF()
    module.ModelTag = "InertialUKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)

    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    rwArrayConfigOut = messaging.RWArrayConfigMsgPayload()
    rwArrayConfigOut.numRW = 3
    rwConfigInMsg = messaging.RWArrayConfigMsg().write(rwArrayConfigOut)


    rwSpeedIntMsg = messaging.RWSpeedMsgPayload()
    rwSpeedIntMsg.wheelSpeeds = [0.1, 0.01, 0.1]
    rwSpeedIntMsg.wheelThetas = [0.,0.,0.]
    rwSpeedInMsg = messaging.RWSpeedMsg().write(rwSpeedIntMsg)

    stMessage1 = messaging.STAttMsgPayload()
    stMessage1.MRP_BdyInrtl = [0.3, 0.4, 0.5]
    st1InMsg = messaging.STAttMsg()

    stMessage2 = messaging.STAttMsgPayload()
    stMessage2.MRP_BdyInrtl = [0.3, 0.4, 0.5]
    st2InMsg = messaging.STAttMsg()

    #    stateTarget = testVector.tolist()
    #    stateTarget.extend([0.0, 0.0, 0.0])
    #    module.state = [0.7, 0.7, 0.0]
    inertialUKFLog = module.logger(["covar", "state"], testProcessRate*10)
    unitTestSim.AddModelToTask(unitTaskName, inertialUKFLog)

    # make input messages but don't write to them
    gyroInMsg = messaging.AccDataMsg()

    # connect messages
    module.STDatasStruct.STMessages[0].stInMsg.subscribeTo(st1InMsg)
    module.STDatasStruct.STMessages[1].stInMsg.subscribeTo(st2InMsg)
    module.massPropsInMsg.subscribeTo(vcInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.gyrBuffInMsg.subscribeTo(gyroInMsg)

    unitTestSim.InitializeSimulation()

    for i in range(20000):
        if i > 20:
            stMessage1.timeTag = int(i * 0.5 * 1E9)
            stMessage2.timeTag = int(i * 0.5 * 1E9)
            st1InMsg.write(stMessage1, unitTestSim.TotalSim.CurrentNanos)
            st2InMsg.write(stMessage2, unitTestSim.TotalSim.CurrentNanos)

        if i==10000:
            rwSpeedIntMsg.wheelSpeeds = [0.5, 0.1, 0.05]
            rwSpeedInMsg.write(rwSpeedIntMsg, 0)

        unitTestSim.ConfigureStopTime(macros.sec2nano((i + 1) * 0.5))
        unitTestSim.ExecuteSimulation()

    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    stateLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.state)
    print(inertialUKFLog.covar, covarLog)

    accuracy = 1.0E-5
    unitTestSupport.writeTeXSnippet("toleranceValue33", str(accuracy), path)
    for i in range(3):
        if (covarLog[-1, i * 6 + 1 + i] > covarLog[0, i * 6 + 1 + i]):
            testFailCount += 1
            testMessages.append("Covariance update with RW failure")
        if (abs(stateLog[-1, i + 1] - stMessage1.MRP_BdyInrtl[i]) > accuracy):
            print(abs(stateLog[-1, i + 1] - stMessage1.MRP_BdyInrtl[i]))
            testFailCount += 1
            testMessages.append("State update with RW failure")
            unitTestSupport.writeTeXSnippet('passFail33', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail33', textSnippetPassed, path)

    stMessage1.MRP_BdyInrtl = [1.2, 0.0, 0.0]
    stMessage2.MRP_BdyInrtl = [1.2, 0.0, 0.0]

    for i in range(20000):
        if i > 20:
            stMessage1.timeTag = int((i + 20000) * 0.25 * 1E9)
            stMessage2.timeTag = int((i + 20000) * 0.5 * 1E9)
            st1InMsg.write(stMessage1, unitTestSim.TotalSim.CurrentNanos)
            st2InMsg.write(stMessage2, unitTestSim.TotalSim.CurrentNanos)

        unitTestSim.ConfigureStopTime(macros.sec2nano((i + 20000 + 1) * 0.5))
        unitTestSim.ExecuteSimulation()

    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    stateLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.state)
    for i in range(3):
        if (covarLog[-1, i * 6 + 1 + i] > covarLog[0, i * 6 + 1 + i]):
            testFailCount += 1
            testMessages.append("Covariance update large failure")
            unitTestSupport.writeTeXSnippet('passFail33', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail33', textSnippetPassed, path)
    plt.figure()
    for i in range(module.numStates):
        plt.plot(stateLog[:, 0] * 1.0E-9, stateLog[:, i + 1], label='State_' +str(i))
        plt.legend()
        plt.ylim([-1, 1])

    unitTestSupport.writeFigureLaTeX('Test31', 'Test 3 State convergence', plt, 'width=0.7\\textwidth, keepaspectratio', path)
    plt.figure()
    for i in range(module.numStates):
        plt.plot(covarLog[:, 0] * 1.0E-9, covarLog[:, i * module.numStates + i + 1], label='Covar_' +str(i))
        plt.legend()
        plt.ylim([0., 2E-7])

    unitTestSupport.writeFigureLaTeX('Test32', 'Test 3 Covariance convergence', plt, 'width=0.7\\textwidth, keepaspectratio', path)
    if (show_plots):
        plt.show()
        plt.close('all')

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state update with RW")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def BROKENtest_StatePropRateInertialAttitude(show_plots):
    [testResults, testMessage] = statePropRateInertialAttitude(show_plots)
    assert testResults < 1, testMessage
def statePropRateInertialAttitude(show_plots):
    """Module Unit Test"""

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
    module = inertialUKF.inertialUKF()
    module.ModelTag = "InertialUKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    module.alpha = 0.02
    module.beta = 2.0
    module.kappa = 0.0
    module.switchMag = 1.2

    module.stateInit = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    module.covarInit = [0.04, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.04, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.04, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.004, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.004, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.004]
    qNoiseIn = numpy.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3] * 0.0017 * 0.0017
    qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6] * 0.00017 * 0.00017
    module.qNoise = qNoiseIn.reshape(36).tolist()

    ST1Data = inertialUKF.STMessage()
    ST1Data.noise = [0.00017 * 0.00017, 0.0, 0.0,
                         0.0, 0.00017 * 0.00017, 0.0,
                         0.0, 0.0, 0.00017 * 0.00017]
    STList = [ST1Data]
    module.STDatasStruct.STMessages = STList
    module.STDatasStruct.numST = len(STList)

    lpDataUse = inertialUKF.LowPassFilterData()
    lpDataUse.hStep = 0.5
    lpDataUse.omegCutoff = 15.0 / (2.0 * math.pi)
    module.gyroFilt = [lpDataUse, lpDataUse, lpDataUse]

    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    stateInit = [0.0, 0.0, 0.0, math.pi/18.0, 0.0, 0.0]
    module.stateInit = stateInit
    inertialUKFLog = module.logger(["covar", "sigma_BNOut", "omega_BN_BOut"], testProcessRate*10)

    stMessage1 = messaging.STAttMsgPayload()
    stMessage1.MRP_BdyInrtl = [0., 0., 0.]
    stMessage1.timeTag = int(1* 1E9)
    st1InMsg = messaging.STAttMsg()

    # make input messages but don't write to them
    rwSpeedInMsg = messaging.RWSpeedMsg()
    rwConfigInMsg = messaging.RWArrayConfigMsg()
    gyroInMsg = messaging.AccDataMsg()

    # connect messages
    module.STDatasStruct.STMessages[0].stInMsg.subscribeTo(st1InMsg)
    module.massPropsInMsg.subscribeTo(vcInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.gyrBuffInMsg.subscribeTo(gyroInMsg)

    unitTestSim.InitializeSimulation()
    st1InMsg.write(stMessage1, int(1 * 1E9))
    gyroBufferData = messaging.AccDataMsgPayload()
    for i in range(3600*2+1):
        gyroBufferData.accPkts[i%inertialUKF.MAX_ACC_BUF_PKT].measTime = (int(i*0.5*1E9))
        gyroBufferData.accPkts[i%inertialUKF.MAX_ACC_BUF_PKT].gyro_B = \
            [math.pi/18.0, 0.0, 0.0]
        gyroInMsg.write(gyroBufferData, (int(i*0.5*1E9)))

        unitTestSim.ConfigureStopTime(macros.sec2nano((i+1)*0.5))
        unitTestSim.ExecuteSimulation()

    covarLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.covar)
    sigmaLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.sigma_BNOut)
    omegaLog = unitTestSupport.addTimeColumn(inertialUKFLog.times(), inertialUKFLog.omega_BN_BOut)
    accuracy = 1.0E-3
    unitTestSupport.writeTeXSnippet("toleranceValue44", str(accuracy), path)
    for i in range(3):
        if(abs(omegaLog[-1, i+1] - stateInit[i+3]) > accuracy):
            print(abs(omegaLog[-1, i+1] - stateInit[i+3]))
            testFailCount += 1
            testMessages.append("State omega propagation failure")
            unitTestSupport.writeTeXSnippet('passFail44', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail44', textSnippetPassed, path)

    for i in range(6):
       if(covarLog[-1, i*6+i+1] <= covarLog[0, i*6+i+1]):
           testFailCount += 1
           testMessages.append("State covariance failure")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state rate propagation")
    else:
        print("Failed: " + testMessages[0])

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def BROKENtest_FaultScenarios(show_plots):
    [testResults, testMessage] = faultScenarios(show_plots)
    assert testResults < 1, testMessage
def faultScenarios(show_plots):
    """Module Unit Test"""
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
    moduleConfigClean1 = inertialUKF.InertialUKFConfig()
    moduleConfigClean1.numStates = 6
    moduleConfigClean1.state = [0., 0., 0., 0., 0., 0.]
    moduleConfigClean1.statePrev = [0., 0., 0., 0., 0., 0.]
    moduleConfigClean1.sBar = [0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 0., 0., 0.]
    moduleConfigClean1.sBarPrev = [1., 0., 0., 0., 0., 0.,
                                   0., 1., 0., 0., 0., 0.,
                                   0., 0., 1., 0., 0., 0.,
                                   0., 0., 0., 1., 0., 0.,
                                   0., 0., 0., 0., 1., 0.,
                                   0., 0., 0., 0., 0., 1.]
    moduleConfigClean1.covar = [0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.,
                                0., 0., 0., 0., 0., 0.]
    moduleConfigClean1.covarPrev = [2., 0., 0., 0., 0., 0.,
                                    0., 2., 0., 0., 0., 0.,
                                    0., 0., 2., 0., 0., 0.,
                                    0., 0., 0., 2., 0., 0.,
                                    0., 0., 0., 0., 2., 0.,
                                    0., 0., 0., 0., 0., 2.]

    inertialUKF.inertialUKFCleanUpdate(moduleConfigClean1)

    if numpy.linalg.norm(numpy.array(moduleConfigClean1.covarPrev) - numpy.array(moduleConfigClean1.covar)) > 1E10:
        testFailCount += 1
        testMessages.append("inertialUKFClean Covar failed")
    if numpy.linalg.norm(numpy.array(moduleConfigClean1.statePrev) - numpy.array(moduleConfigClean1.state)) > 1E10:
        testFailCount += 1
        testMessages.append("inertialUKFClean States failed")
    if numpy.linalg.norm(numpy.array(moduleConfigClean1.sBar) - numpy.array(moduleConfigClean1.sBarPrev)) > 1E10:
        testFailCount += 1
        testMessages.append("inertialUKFClean sBar failed")

    # inertialStateProp rate test with time step difference
    moduleConfigClean1.rwConfigParams.numRW = 2
    moduleConfigClean1.rwSpeeds.wheelSpeeds = [10, 5]
    moduleConfigClean1.rwSpeedPrev.wheelSpeeds = [15, 10]
    moduleConfigClean1.rwConfigParams.JsList = [1., 1.]
    moduleConfigClean1.rwConfigParams.GsMatrix_B = [1., 0., 0., 1., 0., 0.]
    moduleConfigClean1.speedDt = 1.
    #moduleConfigClean1.IInv = [1., 0., 0., 0., 1., 0., 0., 0., 1.]

    # Bad Time and Measurement Update
    st1 = messaging.STAttMsgPayload()
    st1.timeTag = macros.sec2nano(1.)
    st1.MRP_BdyInrtl = [0.1, 0.2, 0.3]

    ST1Data = inertialUKF.STMessage()
    ST1Data.noise = [1., 0., 0.,
                     0., 1., 0.,
                     0., 0., 1.]

    STList = [ST1Data]

    moduleConfigClean1.alpha = 0.02
    moduleConfigClean1.beta = 2.0
    moduleConfigClean1.kappa = 0.0
    moduleConfigClean1.switchMag = 1.2

    moduleConfigClean1.countHalfSPs = moduleConfigClean1.numStates
    moduleConfigClean1.STDatasStruct.STMessages = STList
    moduleConfigClean1.STDatasStruct.numST = len(STList)
    moduleConfigClean1.wC = [-1] * (moduleConfigClean1.numStates * 2 + 1)
    moduleConfigClean1.wM = [-1] * (moduleConfigClean1.numStates * 2 + 1)
    retTime = inertialUKF.inertialUKFTimeUpdate(moduleConfigClean1, 1)
    retMease = inertialUKF.inertialUKFMeasUpdate(moduleConfigClean1, 1)
    if retTime == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Time update")
    if retMease == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Meas update")
    moduleConfigClean1.wC = [1] * (moduleConfigClean1.numStates * 2 + 1)
    moduleConfigClean1.wM = [1] * (moduleConfigClean1.numStates * 2 + 1)
    qNoiseIn = numpy.identity(6)
    qNoiseIn[0:3, 0:3] = -qNoiseIn[0:3, 0:3] * 0.0017 * 0.0017
    qNoiseIn[3:6, 3:6] = -qNoiseIn[3:6, 3:6] * 0.00017 * 0.00017
    moduleConfigClean1.qNoise = qNoiseIn.reshape(36).tolist()
    retTime = inertialUKF.inertialUKFTimeUpdate(moduleConfigClean1, 1)
    retMease = inertialUKF.inertialUKFMeasUpdate(moduleConfigClean1, 1)

    if retTime == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Time update")
    if retMease == 0:
        testFailCount += 1
        testMessages.append("Failed to catch bad Update and clean in Meas update")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: state rate propagation")
    else:
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


if __name__ == "__main__":
    # filterMethods()
    # stateUpdateInertialAttitude(True)
    # statePropInertialAttitude(True)       # Broken test
    # stateUpdateRWInertialAttitude(True)
    # statePropRateInertialAttitude(True)
    # faultScenarios(True)
    all_inertial_kfTest(True)