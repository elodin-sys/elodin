
# ISC License
#
# Copyright (c) 2016-2018, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import relativeODuKF  # import the module that is to be tested
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion

import relativeODuKF_test_utilities as FilterPlots


def addTimeColumn(time, data):
    return np.transpose(np.vstack([[time], np.transpose(data)]))

def rk4(f, t, x0):
    x = np.zeros([len(t),len(x0)+1])
    h = (t[len(t)-1] - t[0])/len(t)
    x[0,0] = t[0]
    x[0,1:] = x0
    for i in range(len(t)-1):
        h = t[i+1] - t[i]
        x[i,0] = t[i]
        k1 = h * f(t[i], x[i,1:])
        k2 = h * f(t[i] + 0.5 * h, x[i,1:] + 0.5 * k1)
        k3 = h * f(t[i] + 0.5 * h, x[i,1:] + 0.5 * k2)
        k4 = h * f(t[i] + h, x[i,1:] + k3)
        x[i+1,1:] = x[i,1:] + (k1 + 2.*k2 + 2.*k3 + k4) / 6.
        x[i+1,0] = t[i+1]
    return x

def twoBodyGrav(t, x, mu = 42828.314*1E9):
    dxdt = np.zeros(np.shape(x))
    dxdt[0:3] = x[3:]
    dxdt[3:] = -mu/np.linalg.norm(x[0:3])**3.*x[0:3]
    return dxdt


def setupFilterData(filterObject):

    filterObject.planetIdInit = 2
    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0

    mu = 42828.314*1E9 #m^3/s^2
    elementsInit = orbitalMotion.ClassicElements()
    elementsInit.a = 4000*1E3 #m
    elementsInit.e = 0.2
    elementsInit.i = 10
    elementsInit.Omega = 0.001
    elementsInit.omega = 0.01
    elementsInit.f = 0.1
    r, v = orbitalMotion.elem2rv(mu, elementsInit)

    filterObject.stateInit = r.tolist() + v.tolist()
    filterObject.covarInit = [1000.*1E6, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 1000.*1E6, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 1000.*1E6, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 5*1E6, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 5*1E6, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 5*1E6]

    qNoiseIn = np.identity(6)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3]*0.00001*0.00001*1E-6
    qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6]*0.0001*0.0001*1E-6
    filterObject.qNoise = qNoiseIn.reshape(36).tolist()
    filterObject.noiseSF = 1

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_

def test_methods_kf(show_plots):
    """Module Unit Test"""
    [testResults, testMessage] = relOD_method_test(show_plots)
    assert testResults < 1, testMessage
def test_propagation_kf(show_plots):
    """Module Unit Test"""
    [testResults, testMessage] = StatePropRelOD(show_plots, 10.0)
    assert testResults < 1, testMessage
def test_measurements_kf(show_plots):
    """Module Unit Test"""
    [testResults, testMessage] = StateUpdateRelOD(show_plots)
    assert testResults < 1, testMessage


def relOD_method_test(show_plots):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    state = [250, 32000, 1000, 5, 3, 2]
    dt = 10
    mu = 42828.314
    # Measurement Model Test
    data = relativeODuKF.RelODuKFConfig()
    msg = messaging.OpNavMsgPayload()
    msg.r_BN_N = [300, 200, 100]
    data.planetId = 2
    data.opNavInBuffer = msg
    data.countHalfSPs = 6
    data.noiseSF = 1

    Covar = np.eye(6)
    SPexp = np.zeros([6, 2*6+1])
    SPexp[:,0] = np.array(state)
    for i in range(1, 6+1):
        SPexp[:,i] = np.array(state) + Covar[:,i-1]
        SPexp[:, i+6] = np.array(state) - Covar[:,i-1]


    data.SP =  np.transpose(SPexp).flatten().tolist()
    relativeODuKF.relODuKFMeasModel(data)

    measurements = data.yMeas

    if np.linalg.norm(np.array(measurements) - np.transpose(SPexp[0:3,:]).flatten()) > 1.0E-15:
        testFailCount += 1
        testMessages.append("Measurement Model Failure")

    # Dynamics Model Test
    data.planetId = 2

    stateIn = relativeODuKF.new_doubleArray(6)
    for i in range(len(state)):
        relativeODuKF.doubleArray_setitem(stateIn, i, state[i])

    relativeODuKF.relODStateProp(data, stateIn, dt)

    propedState = []
    for i in range(6):
        propedState.append(relativeODuKF.doubleArray_getitem(stateIn, i))
    expected = rk4(twoBodyGrav, [0, dt], np.array(state)*1E3)
    expected[:,1:]*=1E-3
    if np.linalg.norm((np.array(propedState) - expected[-1,1:])/(expected[-1,1:])) > 1.0E-15:
        testFailCount += 1
        testMessages.append("State Prop Failure")

    if testFailCount:
        print(testMessages)
    else:
        print("Passed")
    return [testFailCount, ''.join(testMessages)]


def StateUpdateRelOD(show_plots):
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
    dt = 1.0
    t1 = 250
    multT1 = 8

    testProcessRate = macros.sec2nano(dt)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm
    module = relativeODuKF.relativeODuKF()
    module.ModelTag = "relodSuKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)
    module.noiseSF = 1

    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    time = np.linspace(0, int(multT1*t1), int(multT1*t1//dt)+1)
    dydt = np.zeros(6)
    energy = np.zeros(len(time))
    expected=np.zeros([len(time), 7])
    expected[0,1:] = module.stateInit
    mu = 42828.314*1E9
    energy[0] = -mu/(2*orbitalMotion.rv2elem(mu, expected[0,1:4], expected[0,4:]).a)

    kick = np.array([0., 0., 0., -0.01, 0.01, 0.02]) * 10 *1E3

    expected[0:t1,:] = rk4(twoBodyGrav, time[0:t1], module.stateInit)
    expected[t1:multT1*t1+1, :] = rk4(twoBodyGrav, time[t1:len(time)], expected[t1-1, 1:] + kick)
    for i in range(1, len(time)):
        energy[i] = - mu / (2 * orbitalMotion.rv2elem(mu, expected[i, 1:4], expected[i, 4:]).a)

    inputData = messaging.OpNavMsgPayload()
    opnavInMsg = messaging.OpNavMsg()
    module.opNavInMsg.subscribeTo(opnavInMsg)

    inputData.planetID = 2
    inputData.r_BN_B = expected[0, 1:4]

    opnavInMsg.write(inputData, 0)

    unitTestSim.InitializeSimulation()
    for i in range(t1):
        if i > 0 and i % 50 == 0:
            inputData.timeTag = macros.sec2nano(i * dt)
            inputData.r_BN_N = expected[i,1:4] + np.random.normal(0, 5*1E-2, 3)
            inputData.valid = 1
            inputData.covar_N = [5.*1E-2, 0., 0.,
                                 0., 5.*1E-2, 0.,
                                 0., 0., 5.*1E-2]
            opnavInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i + 1) * dt))
        unitTestSim.ExecuteSimulation()

    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    for i in range(6):
        if (covarLog[t1, i * 6 + 1 + i] > covarLog[0, i * 6 + 1 + i] / 100):
            testFailCount += 1
            testMessages.append("Covariance update failure at " + str(t1))

    for i in range(t1, multT1*t1):
        if i % 50 == 0:
            inputData.timeTag = macros.sec2nano(i * dt + 1)
            inputData.r_BN_N = expected[i,1:4] +  np.random.normal(0, 5*1E-2, 3)
            inputData.valid = 1
            inputData.covar_N = [5.*1E-2, 0.,0.,
                                 0., 5.*1E-2, 0.,
                                 0., 0., 5.*1E-2]
            opnavInMsg.write(inputData, unitTestSim.TotalSim.CurrentNanos)
        unitTestSim.ConfigureStopTime(macros.sec2nano((i + 1)*dt))
        unitTestSim.ExecuteSimulation()

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    stateErrorLog = addTimeColumn(dataLog.times(), dataLog.stateError)
    postFitLog = addTimeColumn(dataLog.times(), dataLog.postFitRes)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)


    diff = np.copy(stateLog)
    diff[:,1:]-=expected[:,1:]
    FilterPlots.EnergyPlot(time, energy, 'Update', show_plots)
    FilterPlots.StateCovarPlot(stateLog, covarLog, 'Update', show_plots)
    FilterPlots.StatePlot(diff, 'Update', show_plots)
    FilterPlots.plot_TwoOrbits(expected[:,0:4], stateLog[:,0:4])
    FilterPlots.PostFitResiduals(postFitLog, np.sqrt(5*1E-2*1E6), 'Update', show_plots)

    for i in range(6):
        if (covarLog[t1*multT1, i * 6 + 1 + i] > covarLog[0, i * 6 + 1 + i] / 100):
            testFailCount += 1
            testMessages.append("Covariance update failure at " + str(t1*multT1))

    if (np.linalg.norm(diff[-1, 1:]/expected[-1,1:]) > 1.0E-1):
        testFailCount += 1
        testMessages.append("State propagation failure")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state update")
    else:
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


def StatePropRelOD(show_plots, dt):
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
    testProcessRate = macros.sec2nano(dt)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = relativeODuKF.relativeODuKF()
    module.ModelTag = "relodSuKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    setupFilterData(module)
    module.noiseSF = 1

    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    inputData = messaging.OpNavMsgPayload()
    opnavInMsg = messaging.OpNavMsg()
    module.opNavInMsg.subscribeTo(opnavInMsg)

    opnavInMsg.write(inputData, 0)

    timeSim = 60
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.min2nano(timeSim))
    unitTestSim.ExecuteSimulation()

    time = np.linspace(0, int(timeSim*60), int(timeSim*60//dt)+1)
    dydt = np.zeros(6)
    energy = np.zeros(len(time))
    expected=np.zeros([len(time), 7])
    expected[0,1:] = module.stateInit
    mu = 42828.314*1E9
    energy[0] = -mu/(2*orbitalMotion.rv2elem(mu, expected[0,1:4], expected[0,4:]).a)
    expected = rk4(twoBodyGrav, time, module.stateInit)
    for i in range(1, len(time)):
        energy[i] = - mu / (2 * orbitalMotion.rv2elem(mu, expected[i, 1:4], expected[i, 4:]).a)

    stateLog = addTimeColumn(dataLog.times(), dataLog.state)
    covarLog = addTimeColumn(dataLog.times(), dataLog.covar)

    diff = np.copy(stateLog)
    diff[:,1:]-=expected[:,1:]
    FilterPlots.plot_TwoOrbits(expected[:,0:4], stateLog[:,0:4])
    FilterPlots.EnergyPlot(time, energy, 'Prop', show_plots)
    FilterPlots.StateCovarPlot(stateLog, covarLog, 'Prop', show_plots)
    FilterPlots.StatePlot(diff, 'Prop', show_plots)

    if (np.linalg.norm(diff[-1,1:]/expected[-1,1:]) > 1.0E-10):
        testFailCount += 1
        testMessages.append("State propagation failure")

    if (energy[0] - energy[-1])/energy[0] > 1.0E-10:
        testFailCount += 1
        testMessages.append("State propagation failure")

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag + " state propagation")

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


if __name__ == "__main__":
    # relOD_method_test(True)
    # StatePropRelOD(True, 1.0)
    StateUpdateRelOD(False)
