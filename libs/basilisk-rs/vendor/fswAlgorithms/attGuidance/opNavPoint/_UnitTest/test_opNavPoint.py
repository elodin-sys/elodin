#
# ISC License
#
# Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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


#
#   Copy of the unit test for sunSafe Point adapted to any heading
#   Module Name:        opNavPoint
#   Author:             Thibaud Teil
#   Creation Date:      August 20, 2019
#

import inspect
import os

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))



# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.fswAlgorithms import opNavPoint                   # import the module that is to be tested
from Basilisk.architecture import messaging
from Basilisk.utilities import macros as mc


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
#@pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("case", [
     (1)        # target is visible, vectors are not aligned
    ,(2)        # target is not visible, vectors are not aligned
    ,(3)        # target is visible, vectors are aligned
    ,(4)        # target is not visible, search
])

def test_module(show_plots, case):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = opNavPointTestFunction(show_plots, case)
    assert testResults < 1, testMessage


def opNavPointTestFunction(show_plots, case):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = mc.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    # Construct algorithm and associated C++ container
    module = opNavPoint.opNavPoint()
    module.ModelTag = "opNavPoint"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    camera_Z = [0.,0.,1.]
    module.alignAxis_C = camera_Z
    module.minUnitMag = 0.01
    module.smallAngle = 0.01*mc.D2R
    module.timeOut = 100

    # Create input messages
    #
    planet_B = [1.,1.,0.]
    inputOpNavData = messaging.OpNavMsgPayload()  # Create a structure for the input message
    inputOpNavData.r_BN_C = planet_B
    inputOpNavData.valid = 1
    if (case == 2): #No valid measurement
        inputOpNavData.valid = 0
    if (case == 3): #No valid measurement
        inputOpNavData.r_BN_C = [0.,0.,-1.]
    if (case == 4): #No valid measurement
        inputOpNavData.valid = 0
    opnavInMsg = messaging.OpNavMsg().write(inputOpNavData)

    inputIMUData = messaging.NavAttMsgPayload()  # Create a structure for the input message
    omega_BN_B = np.array([0.01, 0.50, -0.2])
    inputIMUData.omega_BN_B = omega_BN_B
    imuInMsg = messaging.NavAttMsg().write(inputIMUData)
    omega_RN_B_Search = np.array([0.0, 0.0, 0.1])
    if (case ==2 or case==4):
        module.omega_RN_B = omega_RN_B_Search

    cam = messaging.CameraConfigMsgPayload()  # Create a structure for the input message
    cam.sigma_CB = [0.,0.,0]
    camInMsg = messaging.CameraConfigMsg().write(cam)


    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attGuidanceOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.opnavDataInMsg.subscribeTo(opnavInMsg)
    module.imuInMsg.subscribeTo(imuInMsg)
    module.cameraConfigInMsg.subscribeTo(camInMsg)


    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(mc.sec2nano(1.))  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    #
    # check sigma_BR
    #
    # set the filtered output truth states

    eHat = np.cross(-np.array(planet_B), np.array(camera_Z))
    eHat = eHat / np.linalg.norm(eHat)
    Phi = np.arccos(np.dot(-np.array(planet_B)/np.linalg.norm(-np.array(planet_B)),np.array(camera_Z)))
    sigmaTrue = eHat * np.tan(Phi/4.0)
    trueVector = [
                sigmaTrue.tolist(),
                sigmaTrue.tolist(),
                sigmaTrue.tolist()
               ]
    if (case == 2 or case == 3 or case == 4):
        trueVector = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
    # compare the module results to the truth values
    accuracy = 1e-12
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)

    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.sigma_BR[i],trueVector[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_BR unit test at t=" +
                                str(dataLog.times()[i] * mc.NANO2SEC) +
                                "sec\n")

    #
    # check omega_BR_B
    #
    # set the filtered output truth states
    trueVector = [
        omega_BN_B.tolist(),
        omega_BN_B.tolist(),
        omega_BN_B.tolist()
    ]
    if (case == 2 or case==4):
        trueVector = [
            (omega_BN_B - omega_RN_B_Search).tolist(),
            (omega_BN_B - omega_RN_B_Search).tolist(),
            (omega_BN_B - omega_RN_B_Search).tolist()
        ]
    # compare the module results to the truth values
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.omega_BR_B[i],trueVector[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_BR_B unit test at t=" +
                                str(dataLog.times()[i] * mc.NANO2SEC) +
                                "sec\n")
    #
    # check omega_RN_B
    #
    # set the filtered output truth states
    trueVector = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]
    if (case == 2 or case == 4):
        trueVector = [
            omega_RN_B_Search,
            omega_RN_B_Search,
            omega_RN_B_Search
        ]
    # compare the module results to the truth values
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.omega_RN_B[i],trueVector[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_RN_B unit test at t=" +
                                str(dataLog.times()[i] * mc.NANO2SEC) +
                                "sec\n")

    #
    # check domega_RN_B
    #
    # set the filtered output truth states
    trueVector = [
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0]
               ]

    # compare the module results to the truth values
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.domega_RN_B[i],trueVector[i],3,accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed domega_RN_B unit test at t=" +
                                str(dataLog.times()[i] * mc.NANO2SEC) +
                                "sec\n")

    #   print out success message if no error were found
    snippentName = "passFail" + str(case)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("FAILED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)



    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    opNavPointTestFunction(False, 1)
