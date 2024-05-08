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
#   Module Name:        thrMomentumManagement
#   Author:             Hanspeter Schaub
#   Creation Date:      August 18, 2016
#

import inspect
import os

import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))








# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.fswAlgorithms import thrMomentumManagement            # import the module that is to be tested
from Basilisk.utilities import macros
from Basilisk.utilities import fswSetupRW
from Basilisk.architecture import messaging


# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("hsMinCheck", [
    (0),
    (1)
])

# update "module" in this function name to reflect the module name
def test_thrMomentumManagement(show_plots, hsMinCheck):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = thrMomentumManagementTestFunction(show_plots, hsMinCheck)
    assert testResults < 1, testMessage


def thrMomentumManagementTestFunction(show_plots, hsMinCheck):
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
    module = thrMomentumManagement.thrMomentumManagement()
    module.ModelTag = "thrMomentumManagement"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    if hsMinCheck:
        module.hs_min = 1000./6000.*100.               # Nms
    else:
        module.hs_min = 100./6000.*100.               # Nms


    # wheelSpeeds Message
    rwSpeedMessage = messaging.RWSpeedMsgPayload()
    rwSpeedMessage.wheelSpeeds = [10.0, -25.0, 50.0, 100.]
    rwSpeedInMsg = messaging.RWSpeedMsg().write(rwSpeedMessage)


    # wheelConfigData Message
    fswSetupRW.clearSetup()
    Js = 0.1
    fswSetupRW.create([1.0, 0.0, 0.0], Js)
    fswSetupRW.create([0.0, 1.0, 0.0], Js)
    fswSetupRW.create([0.0, 0.0, 1.0], Js)
    fswSetupRW.create([0.5773502691896258, 0.5773502691896258, 0.5773502691896258], Js)
    rwConfigInMsg = fswSetupRW.writeConfigMessage()



    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.deltaHOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # setup message connections
    module.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    module.rwConfigDataInMsg.subscribeTo(rwConfigInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.5))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # set the filtered output truth states
    if hsMinCheck == 1:
        trueVector = [
                   [0.0, 0.0, 0.0]
                   ]*2
    else:
        trueVector = [
                   [-5.914369484146579, -2.858300248464629, -9.407020039211664]
                   ]*2

    # compare the module results to the truth values
    accuracy = 1e-12
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, dataLog.torqueRequestBody, accuracy,
                                                               "torqueRequestBody", testFailCount, testMessages)

    snippetName = "passFail" + str(hsMinCheck)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippetName, passedText, path)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_thrMomentumManagement(              # update "module" in function name
                 True,
                 0            # hsMinCheck
               )
