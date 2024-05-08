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
#   Module Name:        mrpRotation
#   Author:             Hanspeter Schaub
#   Creation Date:      May 20, 2018
#

import inspect
import os
import sys

import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

import numpy as np


# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.fswAlgorithms import mrpRotation                    # import the module that is to be tested
from Basilisk.utilities import macros as mc
from Basilisk.architecture import messaging


sys.path.append(path + '/Support')
import truth_mrpRotation as truth


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)

@pytest.mark.parametrize("cmdStateFlag", [False, True])
@pytest.mark.parametrize("testReset", [False, True])



# provide a unique test method name, starting with test_
def test_mrpRotation(show_plots, cmdStateFlag, testReset):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = run(show_plots, cmdStateFlag, testReset)
    assert testResults < 1, testMessage


def run(show_plots, cmdStateFlag, testReset):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)


    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Test times
    updateTime = 0.5     # update process rate update time
    totalTestSimTime = 1.5

    # Create test thread
    testProcessRate = mc.sec2nano(updateTime)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    # Construct algorithm and associated C++ container
    module = mrpRotation.mrpRotation()
    module.ModelTag = "mrpRotation"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    sigma_RR0 = np.array([0.3, .5, 0.0])
    module.mrpSet = sigma_RR0
    omega_RR0_R = np.array([0.1, 0.0, 0.0]) * mc.D2R
    module.omega_RR0_R = omega_RR0_R
    unitTestSupport.writeTeXSnippet("sigma_RR0", str(sigma_RR0), path)
    unitTestSupport.writeTeXSnippet("omega_RR0_R", str(omega_RR0_R*mc.R2D) + "deg/sec", path)


    if cmdStateFlag:
        desiredAtt = messaging.AttStateMsgPayload()
        sigma_RR0 = np.array([0.1, 0.0, -0.2])
        desiredAtt.state = sigma_RR0
        omega_RR0_R = np.array([0.1, 1.0, 0.5]) * mc.D2R
        desiredAtt.rate = omega_RR0_R
        desInMsg = messaging.AttStateMsg().write(desiredAtt)
        module.desiredAttInMsg.subscribeTo(desInMsg)

        unitTestSupport.writeTeXSnippet("sigma_RR0Cmd", str(sigma_RR0), path)
        unitTestSupport.writeTeXSnippet("omega_RR0_RCmd", str(omega_RR0_R * mc.R2D) + "deg/sec", path)


    #
    # Reference Frame Message
    #
    RefStateInData = messaging.AttRefMsgPayload()  # Create a structure for the input message
    sigma_R0N = np.array([0.1, 0.2, 0.3])
    RefStateInData.sigma_RN = sigma_R0N
    omega_R0N_N = np.array([0.1, 0.0, 0.0])
    RefStateInData.omega_RN_N = omega_R0N_N
    domega_R0N_N = np.array([0.0, 0.0, 0.0])
    RefStateInData.domega_RN_N = domega_R0N_N
    attRefMsg = messaging.AttRefMsg().write(RefStateInData)
    module.attRefInMsg.subscribeTo(attRefMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(mc.sec2nano(totalTestSimTime))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    if testReset:
        module.Reset(1)
        unitTestSim.ConfigureStopTime(mc.sec2nano(totalTestSimTime+1.0))        # seconds to stop simulation
        unitTestSim.ExecuteSimulation()


    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    accuracy = 1e-12
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)
    trueSigma, trueOmega, truedOmega, \
        = truth.results(sigma_RR0,omega_RR0_R,RefStateInData,updateTime, cmdStateFlag, testReset)

    #
    # check sigma_RN
    #
    testFailCount, testMessages = unitTestSupport.compareArray(trueSigma, dataLog.sigma_RN,
                                                               accuracy, "sigma_RN Set",
                                                               testFailCount, testMessages)
    #
    # check omega_RN_N
    #
    testFailCount, testMessages = unitTestSupport.compareArray(trueOmega, dataLog.omega_RN_N,
                                                               accuracy, "omega_RN_N Vector",
                                                               testFailCount, testMessages)

    #
    # check domega_RN_N
    #
    testFailCount, testMessages = unitTestSupport.compareArray(truedOmega, dataLog.domega_RN_N,
                                                               accuracy, "domega_RN_N Vector",
                                                               testFailCount, testMessages)


    snippentName = "passFail" + str(cmdStateFlag) + str(testReset)
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
    test_mrpRotation(
        False           # show plots
        , False         # cmdStateFlag
        , True         # testReset
    )
