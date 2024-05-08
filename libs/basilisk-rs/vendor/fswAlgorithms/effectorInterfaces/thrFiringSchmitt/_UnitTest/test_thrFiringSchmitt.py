
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
#   Unit Test Script
#   Module Name:        thrFiringSchmitt
#   Author:             John Alcorn
#   Creation Date:      August 25, 2016
#


import inspect
import os

import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))








# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.fswAlgorithms import thrFiringSchmitt            # import the module that is to be tested
from Basilisk.utilities import macros
from Basilisk.utilities import fswSetupThrusters
from Basilisk.architecture import messaging


# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("resetCheck, dvOn", [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
])

# update "module" in this function name to reflect the module name
def test_thrFiringSchmitt(show_plots, resetCheck, dvOn):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = thrFiringSchmittTestFunction(show_plots, resetCheck, dvOn)
    assert testResults < 1, testMessage


def thrFiringSchmittTestFunction(show_plots, resetCheck, dvOn):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    # Construct algorithm and associated C++ container
    module = thrFiringSchmitt.thrFiringSchmitt()
    module.ModelTag = "thrFiringSchmitt"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.thrMinFireTime = 0.2
    if dvOn == 1:
        module.baseThrustState = 1
    else:
        module.baseThrustState = 0

    module.level_on = .75
    module.level_off = .25

    # setup thruster cluster message
    fswSetupThrusters.clearSetup()
    rcsLocationData = [
        [-0.86360, -0.82550, 1.79070],
        [-0.82550, -0.86360, 1.79070],
        [0.82550, 0.86360, 1.79070],
        [0.86360, 0.82550, 1.79070],
        [-0.86360, -0.82550, -1.79070],
        [-0.82550, -0.86360, -1.79070],
        [0.82550, 0.86360, -1.79070],
        [0.86360, 0.82550, -1.79070]
        ]
    rcsDirectionData = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
        ]

    for i in range(len(rcsLocationData)):
        fswSetupThrusters.create(rcsLocationData[i], rcsDirectionData[i], 0.5)
    thrConfMsg = fswSetupThrusters.writeConfigMessage()
    numThrusters = fswSetupThrusters.getNumOfDevices()
    module.thrConfInMsg.subscribeTo(thrConfMsg)

    # setup thruster impulse request message
    inputMessageData = messaging.THRArrayCmdForceMsgPayload()
    thrCmdMsg = messaging.THRArrayCmdForceMsg()
    module.thrForceInMsg.subscribeTo(thrCmdMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.onTimeOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.

    if dvOn:
        effReq1 = [0.0, -0.1, -0.2, -0.3, -0.349, -0.351, -0.451, -0.5]
        effReq2 = [0.0, -0.1, -0.2, -0.3, -0.351, -0.351, -0.451, -0.5]
        effReq3 = [0.0, -0.1, -0.2, -0.3, -0.5, -0.351, -0.451, -0.5]
        effReq4 = [0.0, -0.1, -0.2, -0.3, -0.351, -0.351, -0.451, -0.5]

    else:
        effReq1 = [0.5, 0.05, 0.09, 0.11, 0.16, 0.18, 0.2, 0.49]
        effReq2 = [0.5, 0.05, 0.09, 0.11, 0.16, 0.18, 0.2, 0.11]
        effReq3 = [0.5, 0.05, 0.09, 0.11, 0.16, 0.18, 0.2, 0.01]
        effReq4 = [0.5, 0.05, 0.09, 0.11, 0.16, 0.18, 0.2, 0.11]

    inputMessageData.thrForce = effReq1
    thrCmdMsg.write(inputMessageData)
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()


    inputMessageData.thrForce = effReq2
    thrCmdMsg.write(inputMessageData)
    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()


    inputMessageData.thrForce = effReq3
    thrCmdMsg.write(inputMessageData)
    unitTestSim.ConfigureStopTime(macros.sec2nano(2.5))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()


    inputMessageData.thrForce = effReq4
    thrCmdMsg.write(inputMessageData)
    unitTestSim.ConfigureStopTime(macros.sec2nano(3.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    if resetCheck:
        # reset the module to test this functionality
        module.Reset(macros.sec2nano(3.0))     # this module reset function needs a time input (in NanoSeconds)

        # run the module again for an additional 1.0 seconds
        unitTestSim.ConfigureStopTime(macros.sec2nano(5.5))        # seconds to stop simulation
        unitTestSim.ExecuteSimulation()


    # This pulls the actual data log from the simulation run.
    moduleOutput = dataLog.OnTimeRequest[:, :numThrusters]
    # print moduleOutput

    # set the filtered output truth states
    if resetCheck==1:
        if dvOn == 1:
            trueVector = [
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   ]
        else:
            trueVector = [
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.49],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   ]

    else:
        if dvOn == 1:
            trueVector = [
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.2, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.4, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0],
                   ]
        else:
            trueVector = [
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.49],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   [0.55, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
                   ]

        # else:
        #     testFailCount+=1
        #     testMessages.append("FAILED: " + module.ModelTag + " Module failed with unsupported input parameters")

    # compare the module results to the truth values
    accuracy = 1e-12
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)

    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput, accuracy,
                                                               "OnTimeRequest", testFailCount, testMessages)

    snippentName = "passFail" + str(resetCheck) + str(dvOn)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
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
    test_thrFiringSchmitt(              # update "module" in function name
                 False,
                 True,           # resetOn
                 False           # dvOn
               )
