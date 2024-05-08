
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
from Basilisk.fswAlgorithms import pixelLineBiasUKF  # import the module that is to be tested
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion

import pixelLineBias_test_utilities as FilterPlots


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
    dxdt[0:3] = x[3:6]
    dxdt[3:6] = -mu/np.linalg.norm(x[0:3])**3.*x[0:3]
    return dxdt


def setupFilterData(filterObject):

    filterObject.planetIdInit = 2
    filterObject.alpha = 0.02
    filterObject.beta = 2.0
    filterObject.kappa = 0.0
    filterObject.gamma = 0.9

    mu = 42828.314*1E9 #m^3/s^2
    elementsInit = orbitalMotion.ClassicElements()
    elementsInit.a = 8000*1E3 #m
    elementsInit.e = 0.2
    elementsInit.i = 10
    elementsInit.Omega = 0.001
    elementsInit.omega = 0.01
    elementsInit.f = 0.1
    r, v = orbitalMotion.elem2rv(mu, elementsInit)
    bias = [1,1,-2]

    filterObject.stateInit = r.tolist() + v.tolist() + bias
    filterObject.covarInit = [1000.*1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 1000.*1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 1000.*1E6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 5*1E6, 0.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 5*1E6, 0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 5*1E6, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,]

    qNoiseIn = np.identity(9)
    qNoiseIn[0:3, 0:3] = qNoiseIn[0:3, 0:3]*0.00001*0.00001*1E-6
    qNoiseIn[3:6, 3:6] = qNoiseIn[3:6, 3:6]*0.0001*0.0001*1E-6
    qNoiseIn[6:9, 6:9] = qNoiseIn[3:6, 3:6]*0.0001*0.0001
    filterObject.qNoise = qNoiseIn.reshape(9*9).tolist()

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


def relOD_method_test(show_plots):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    state = [250, 32000, 1000, 5, 3, 2, 1, 1, 1]
    covar = 10* np.eye(len(state))
    dt = 10
    mu = 42828.314
    # Measurement Model Test
    data = pixelLineBiasUKF.PixelLineBiasUKFConfig()
    msg = messaging.OpNavCirclesMsgPayload()
    msg.circlesCenters = [100, 200]
    msg.circlesRadii = [100]
    msg.planetIds = [2]
    data.circlesInBuffer = msg
    data.planetId = 2
    data.countHalfSPs = len(state)
    data.numStates = len(state)

    # Dynamics Model Test
    data.planetId = 2

    stateIn = pixelLineBiasUKF.new_doubleArray(len(state))
    for i in range(len(state)):
        pixelLineBiasUKF.doubleArray_setitem(stateIn, i, state[i])

    pixelLineBiasUKF.relODStateProp(data, stateIn, dt)

    propedState = []
    for i in range(len(state)):
        propedState.append(pixelLineBiasUKF.doubleArray_getitem(stateIn, i))
    expected = rk4(twoBodyGrav, [0, dt], np.array(state)*1E3)
    expected[:,1:]*=1E-3
    if np.linalg.norm((np.array(propedState) - expected[-1,1:])/(expected[-1,1:])) > 1.0E-15:
        testFailCount += 1
        testMessages.append("State Prop Failure")

    # Set up a measurement test
    data = pixelLineBiasUKF.PixelLineBiasUKFConfig()
    # Set up a circle input message
    msg = messaging.OpNavCirclesMsgPayload()
    msg.circlesCenters = [100, 200]
    msg.circlesRadii = [100]
    msg.planetIds = [2]
    data.circlesInBuffer = msg
    data.planetId = 2
    data.countHalfSPs = len(state)
    data.numStates = len(state)

    # Set up attitud message
    att = messaging.NavAttMsgPayload()
    att.sigma_BN = [0, 0.2,-0.1]
    att.omega_BN_B = [0.,0.,0.]
    data.attInfo = att

    # Set up a camera message
    cam = messaging.CameraConfigMsgPayload()
    cam.sigma_CB = [-0.2, 0., 0.3]
    cam.fieldOfView = 2.0 * np.arctan(10*1e-3 / 2.0 / (1.*1e-3) )  # 2*arctan(s/2 / f)
    cam.resolution = [512, 512]
    data.cameraSpecs = cam

    # Populate sigma points
    SP = np.zeros([len(state), 2*len(state) +1])
    for i in range(2*len(state) + 1):
        if i ==0:
            SP[:, i] = np.array(state)
        if i < len(state) + 1 and i>0:
            SP[:,i] = np.array(state) + covar[:,i-1]
        if i > len(state):
            SP[:,i] = np.array(state) - covar[:,i-(len(state)+1)]

    data.SP = np.transpose(SP).flatten().tolist()
    data.state = state
    pixelLineBiasUKF.pixelLineBiasUKFMeasModel(data)

    yMeasOut = data.yMeas
    expectedMeas = np.zeros([6, 2*len(state)+1])

    dcm_CB = rbk.MRP2C(cam.sigma_CB)
    dcm_BN = rbk.MRP2C(att.sigma_BN)
    dcm_CN = np.dot(dcm_CB, dcm_BN)

    pX = 2. * np.tan(cam.fieldOfView * cam.resolution[0] / cam.resolution[1] / 2.0)
    pY = 2. * np.tan(cam.fieldOfView / 2.0)
    X = pX / cam.resolution[0]
    Y = pY / cam.resolution[1]
    planetRad = 3396.19
    obs = np.array([msg.circlesCenters[0], msg.circlesCenters[1], msg.circlesRadii[0], 0, 0, 0])
    for i in range(2*len(state)+1):
        r_C = np.dot(dcm_CN, SP[0:3,i])
        rNorm = np.linalg.norm(SP[0:3,i])
        r_C = -1./r_C[2]*r_C

        centerX = r_C[0] / X
        centerY = r_C[1] / Y
        centerX += cam.resolution[0]/2 - 0.5
        centerY += cam.resolution[1] / 2 - 0.5

        rad = 1.0/X*np.tan(np.arcsin(planetRad/rNorm))

        if i == 0:
            obs[3:5] = np.array(msg.circlesCenters[0:2]) - obs[0:2]
            obs[5] = rad - obs[2]
        for j in range(3):
            obs[3+j] = round(obs[3+j])
        expectedMeas[0,i] = centerX - SP[6, i]
        expectedMeas[1,i] = centerY - SP[7, i]
        expectedMeas[2, i] = rad - SP[8, i]
        expectedMeas[3:, i] = SP[6:, i]

    yMeasTest = np.zeros([6, 2*len(state)+1])
    for i in range(2*len(state)+1):
        yMeasTest[:,i] = yMeasOut[i*6:i*6+6]
    if np.linalg.norm((yMeasTest - expectedMeas))/np.linalg.norm(expectedMeas[:,0]) > 1.0E-15:
        testFailCount += 1
        testMessages.append("State Prop Failure")

    if testFailCount == 0:
        print("PASSED: ")
    else:
        print(testMessages)

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
    state = [250, 32000, 1000, 5, 3, 2, 1, 1, 1]
    testProcessRate = macros.sec2nano(dt)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = pixelLineBiasUKF.pixelLineBiasUKF()
    module.ModelTag = "relodSuKF"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)
    setupFilterData(module)

    # Create the input messages.
    inputCamera = messaging.CameraConfigMsgPayload()
    inputAtt = messaging.NavAttMsgPayload()

    # Set camera
    inputCamera.fieldOfView = 2.0 * np.arctan(10*1e-3 / 2.0 / 0.01)  # 2*arctan(s/2 / f)
    inputCamera.resolution = [512, 512]
    inputCamera.sigma_CB = [1., 0.3, 0.1]
    camInMsg = messaging.CameraConfigMsg().write(inputCamera)
    module.cameraConfigInMsg.subscribeTo(camInMsg)

    # Set attitude
    inputAtt.sigma_BN = [0.6, 1., 0.1]
    attInMsg = messaging.NavAttMsg().write(inputAtt)
    module.attInMsg.subscribeTo(attInMsg)

    circlesInMsg = messaging.OpNavCirclesMsg()
    module.circlesInMsg.subscribeTo(circlesInMsg)

    dataLog = module.filtDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    timeSim = 60
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.min2nano(timeSim))
    unitTestSim.ExecuteSimulation()

    time = np.linspace(0, timeSim*60, (int) (timeSim*60/dt+1))
    dydt = np.zeros(len(module.stateInit))
    energy = np.zeros(len(time))
    expected=np.zeros([len(time), len(module.stateInit)+1])
    expected[0,1:] = module.stateInit
    mu = 42828.314*1E9
    energy[0] = -mu/(2*orbitalMotion.rv2elem(mu, expected[0,1:4], expected[0,4:7]).a)
    expected = rk4(twoBodyGrav, time, module.stateInit)
    for i in range(1, len(time)):
        energy[i] = - mu / (2 * orbitalMotion.rv2elem(mu, expected[i, 1:4], expected[i, 4:7]).a)

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
    else:
        print(testMessages)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


if __name__ == "__main__":
    # relOD_method_test(True)
    StatePropRelOD(True, 1.0)
