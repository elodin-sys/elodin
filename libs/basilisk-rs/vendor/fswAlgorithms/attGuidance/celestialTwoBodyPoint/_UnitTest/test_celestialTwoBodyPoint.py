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
#

#   Unit Test Script
#   Module Name:        celestialTwoBodyPoint
#   Author:             Mar Cols
#   Creation Date:      May 11, 2016
#

import inspect
import os

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import celestialTwoBodyPoint  # module that is to be tested
from Basilisk.utilities import RigidBodyKinematics as rbk
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import astroFunctions as af
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from numpy import linalg as la

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
textSnippetPassed = r'\textcolor{ForestGreen}{' + "PASSED" + '}'
textSnippetFailed = r'\textcolor{Red}{' + "Failed" + '}'


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

def computeCelestialTwoBodyPoint(R_P1, v_P1, a_P1, R_P2, v_P2, a_P2):

    # Beforehand computations
    R_n = np.cross(R_P1, R_P2)
    v_n = np.cross(v_P1, R_P2) + np.cross(R_P1, v_P2)
    a_n = np.cross(a_P1, R_P2) + np.cross(R_P1, a_P2) + 2 * np.cross(v_P1, v_P2)

    # Reference Frame generation
    r1_hat = R_P1/la.norm(R_P1)
    r3_hat = R_n/la.norm(R_n)
    r2_hat = np.cross(r3_hat, r1_hat)
    RN = np.array([r1_hat, r2_hat, r3_hat])
    sigma_RN = rbk.C2MRP(RN)

    # Reference base-vectors first time-derivative
    I_33 = np.identity(3)
    C1 = I_33 - np.outer(r1_hat, r1_hat)
    dr1_hat = 1.0 / la.norm(R_P1) * np.dot(C1, v_P1)
    C3 = I_33 - np.outer(r3_hat, r3_hat)
    dr3_hat = 1.0 / la.norm(R_n) * np.dot(C3, v_n)
    dr2_hat = np.cross(dr3_hat, r1_hat) + np.cross(r3_hat, dr1_hat)

    # Angular Velocity computation
    omega_RN_R = np.array([
        np.dot(r3_hat, dr2_hat),
        np.dot(r1_hat, dr3_hat),
        np.dot(r2_hat, dr1_hat)
    ])
    omega_RN_N = np.dot(RN.T, omega_RN_R)

    # Reference base-vectors second time-derivative
    temp33_1 = 2 * np.outer(dr1_hat, r1_hat) + np.outer(r1_hat, dr1_hat)
    ddr1_hat = 1.0 / la.norm(R_P1) * (np.dot(C1, a_P1) - np.dot(temp33_1, v_P1))
    temp33_3 = 2 * np.outer(dr3_hat, r3_hat) + np.outer(r3_hat, dr3_hat)
    ddr3_hat = 1.0 / la.norm(R_n) * (np.dot(C3, a_n) - np.dot(temp33_3, v_n))
    ddr2_hat = np.cross(ddr3_hat, r1_hat) + np.cross(ddr1_hat, r3_hat) + 2 * np.cross(dr3_hat, dr1_hat)

    # Angular Acceleration computation
    domega_RN_R = np.array([
        np.dot(dr3_hat, dr2_hat) + np.dot(r3_hat, ddr2_hat) - np.dot(omega_RN_R, dr1_hat),
        np.dot(dr1_hat, dr3_hat) + np.dot(r1_hat, ddr3_hat) - np.dot(omega_RN_R, dr2_hat),
        np.dot(dr2_hat, dr1_hat) + np.dot(r2_hat, ddr1_hat) - np.dot(omega_RN_R, dr3_hat)

    ])
    domega_RN_N = np.dot(RN.T, domega_RN_R)

    return sigma_RN, omega_RN_N, domega_RN_N

def test_celestialTwoBodyPointTestFunction(show_plots):
    """Module Unit Test"""

    [testResults, testMessage] = celestialTwoBodyPointTestFunction(show_plots)
    assert testResults < 1, testMessage



def celestialTwoBodyPointTestFunction(show_plots):

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = celestialTwoBodyPoint.celestialTwoBodyPoint()
    module.ModelTag = "celestialTwoBodyPoint"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.singularityThresh = 1.0 * af.D2R


    # Previous Computation of Initial Conditions for the test
    a = af.E_radius * 2.8
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    f = 60 * af.D2R
    (r, v) = af.OE2RV(af.mu_E, a, e, i, Omega, omega, f)
    r_BN_N = np.array([0., 0., 0.])
    v_BN_N = np.array([0., 0., 0.])
    celPositionVec = r
    celVelocityVec = v

    # Create input message and size it because the regular creator of that message
    # is not part of the test.
    #   Navigation Input Message

    NavStateOutData = messaging.NavTransMsgPayload()  # Create a structure for the input message
    NavStateOutData.r_BN_N = r_BN_N
    NavStateOutData.v_BN_N = v_BN_N
    navMsg = messaging.NavTransMsg().write(NavStateOutData)

    #   Spice Input Message of Primary Body

    CelBodyData = messaging.EphemerisMsgPayload()
    CelBodyData.r_BdyZero_N = celPositionVec
    CelBodyData.v_BdyZero_N = celVelocityVec
    celBodyMsg = messaging.EphemerisMsg().write(CelBodyData)


    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.transNavInMsg.subscribeTo(navMsg)
    module.celBodyInMsg.subscribeTo(celBodyMsg)


    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.))  # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    ## Set truth values
    a = af.E_radius * 2.8
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    f = 60 * af.D2R
    (r, v) = af.OE2RV(af.mu_E, a, e, i, Omega, omega, f)
    r_BN_N = np.array([0., 0., 0.])
    v_BN_N = np.array([0., 0., 0.])
    celPositionVec = r
    celVelocityVec = v

    # Begin Method
    R_P1 = celPositionVec - r_BN_N
    v_P1 = celVelocityVec - v_BN_N
    a_P1 = np.array([0., 0., 0.])
    R_P2 = np.cross(R_P1, v_P1)
    v_P2 = np.cross(R_P1, a_P1)
    a_P2 = np.cross(v_P1, a_P1)

    sigma_RN, omega_RN_N, domega_RN_N = computeCelestialTwoBodyPoint(R_P1, v_P1, a_P1, R_P2, v_P2, a_P2)

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    # check sigma_RN
    moduleOutput = dataLog.sigma_RN
    # compare the module results to the truth values
    accuracy = 1e-12
    # check a vector values
    for i in range(0, len(moduleOutput)):
        if not unitTestSupport.isArrayEqual(moduleOutput[i], sigma_RN, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_RN unit test at t=" +
                                str(moduleOutput[i, 0] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail11', textSnippetPassed, path)

    # check omega_RN_N
    moduleOutput = dataLog.omega_RN_N

    # compare the module results to the truth values
    # check a vector values
    for i in range(0, len(moduleOutput)):
        if not unitTestSupport.isArrayEqual(moduleOutput[i], omega_RN_N, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_RN_N unit test at t=" +
                                str(moduleOutput[i, 0] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail12', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail12', textSnippetPassed, path)

    # check domega_RN_N
    moduleOutput = dataLog.domega_RN_N

    # compare the module results to the truth values
    # check a vector values
    for i in range(0, len(moduleOutput)):
        if not unitTestSupport.isArrayEqual(moduleOutput[i], domega_RN_N, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed domega_RN_N unit test at t=" +
                                str(moduleOutput[i, 0] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail13', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail13', textSnippetPassed, path)

    if testFailCount == 0:
        print("PASSED: " + "celestialTwoBodyPointTestFunction")
    else:
        print(testMessages)

    return [testFailCount, ''.join(testMessages)]

def test_secBodyCelestialTwoBodyPointTestFunction(show_plots):
    """Module Unit Test"""

    # each test method requires a single assert method to be called
    [testResults, testMessage] = secBodyCelestialTwoBodyPointTestFunction(show_plots)
    assert testResults < 1, testMessage

def secBodyCelestialTwoBodyPointTestFunction(show_plots):

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = celestialTwoBodyPoint.celestialTwoBodyPoint()
    module.ModelTag = "secBodyCelestialTwoBodyPoint"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.singularityThresh = 1.0 * af.D2R

    # Previous Computation of Initial Conditions for the test
    a = af.E_radius * 2.8
    e = 0.0
    i = 0.0
    Omega = 0.0
    omega = 0.0
    f = 60 * af.D2R
    (r, v) = af.OE2RV(af.mu_E, a, e, i, Omega, omega, f)
    r_BN_N = np.array([0., 0., 0.])
    v_BN_N = np.array([0., 0., 0.])
    celPositionVec = r
    celVelocityVec = v

    # Create input message and size it because the regular creator of that message

    # is not part of the test.

    #   Navigation Input Message
    NavStateOutData = messaging.NavTransMsgPayload()  # Create a structure for the input message
    NavStateOutData.r_BN_N = r_BN_N
    NavStateOutData.v_BN_N = v_BN_N
    navMsg = messaging.NavTransMsg().write(NavStateOutData)

    #   Spice Input Message of Primary Body
    CelBodyData = messaging.EphemerisMsgPayload()
    CelBodyData.r_BdyZero_N = celPositionVec
    CelBodyData.v_BdyZero_N = celVelocityVec
    celBodyMsg = messaging.EphemerisMsg().write(CelBodyData)

    #   Spice Input Message of Secondary Body
    SecBodyData = messaging.EphemerisMsgPayload()
    secPositionVec = [500., 500., 500.]
    SecBodyData.r_BdyZero_N = secPositionVec
    secVelocityVec = [0., 0., 0.]
    SecBodyData.v_BdyZero_N = secVelocityVec
    cel2ndBodyMsg = messaging.EphemerisMsg().write(SecBodyData)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.transNavInMsg.subscribeTo(navMsg)
    module.celBodyInMsg.subscribeTo(celBodyMsg)
    module.secCelBodyInMsg.subscribeTo(cel2ndBodyMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.))  # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    # check sigma_RN
    moduleOutput = dataLog.sigma_RN

    # set the filtered output truth states
    trueVector = [0.474475084038,  0.273938317493,  0.191443718765]

    # compare the module results to the truth values
    accuracy = 1e-10
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)

    for i in range(0, len(moduleOutput)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(moduleOutput[i], trueVector, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_RN unit test at t=" +
                                str(dataLog.times()[i] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail21', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail21', textSnippetPassed, path)

    # check omega_RN_N
    moduleOutput = dataLog.omega_RN_N

    # set the filtered output truth states
    trueVector = [1.59336987e-04,   2.75979758e-04,   2.64539877e-04]
    # compare the module results to the truth values
    for i in range(0, len(moduleOutput)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(moduleOutput[i], trueVector, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_RN_N unit test at t=" +
                                str(dataLog.times()[i] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail22', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail22', textSnippetPassed, path)

    # check domega_RN_N
    moduleOutput = dataLog.domega_RN_N

    # set the filtered output truth states
    trueVector = [-2.12284893e-07,   5.69968291e-08,  -4.83648052e-08]

    # compare the module results to the truth values
    for i in range(0, len(moduleOutput)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(moduleOutput[i], trueVector, 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed domega_RN_N unit test at t=" +
                                str(dataLog.times()[i] * macros.NANO2SEC) +
                                "sec\n")
            unitTestSupport.writeTeXSnippet('passFail23', textSnippetFailed, path)
        else:
            unitTestSupport.writeTeXSnippet('passFail23', textSnippetPassed, path)

    # Note that we can continue to step the simulation however we feel like.
    # Just because we stop and query data does not mean everything has to stop for good
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.6))  # run an additional 0.6 seconds
    unitTestSim.ExecuteSimulation()

    if testFailCount == 0:
        print("PASSED: " + "secBodyCelestialTwoBodyPointTestFunction")
    else:
        print(testMessages)

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    # celestialTwoBodyPointTestFunction(False)
    secBodyCelestialTwoBodyPointTestFunction(False)
