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
#   Module Name:        eulerRotation
#   Author:             Mar Cols
#   Creation Date:      January 22, 2016
#

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import eulerRotation  # import the module that is to be tested
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros as mc
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

@pytest.mark.parametrize("function", ["run"
                                      , "run2"
                                      ])
def test_all_test_eulerRotation(show_plots, function):
    """Module Unit Test"""
    [testResults, testMessage] = eval(function + '(show_plots)')
    assert testResults < 1, testMessage


def run(show_plots):
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
    module = eulerRotation.eulerRotation()
    module.ModelTag = "eulerRotation"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    angleSet = np.array([0.0, 90.0, 0.0]) * mc.D2R
    module.angleSet = angleSet
    angleRates = np.array([0.1, 0.0, 0.0]) * mc.D2R
    module.angleRates = angleRates

    # Create input message and size it because the regular creator of that message
    # is not part of the test.

    #
    # Reference Frame Message
    #
    RefStateOutData = messaging.AttRefMsgPayload()  # Create a structure for the input message
    sigma_R0N = np.array([0.1, 0.2, 0.3])
    RefStateOutData.sigma_RN = sigma_R0N
    omega_R0N_N = np.array([0.1, 0.0, 0.0])
    RefStateOutData.omega_RN_N = omega_R0N_N
    domega_R0N_N = np.array([0.0, 0.0, 0.0])
    RefStateOutData.domega_RN_N = domega_R0N_N
    attRefInMsg = messaging.AttRefMsg().write(RefStateOutData)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.attRefInMsg.subscribeTo(attRefInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(mc.sec2nano(totalTestSimTime))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    accuracy = 1e-12
    #
    # check sigma_RN
    #
    moduleOutput = dataLog.sigma_RN

    # set the filtered output truth states
    trueVector = [
        [-0.193031238249, 0.608048400483, 0.386062476497],
        [-0.193031238249, 0.608048400483, 0.386062476497],
        [-0.193144351314,  0.607931107381,  0.386360300559],
        [-0.193257454832,  0.607813704445,  0.386658117585]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "sigma_RN Set",
                                                               testFailCount, testMessages)
    # print '\n sigma_RN = ', moduleOutput[:, 1:], '\n'
    #
    # check omega_RN_N
    #
    moduleOutput = dataLog.omega_RN_N
    # set the filtered output truth states
    trueVector = [
        [0.101246280045,  0.000182644489,  0.001208139578],
        [0.101246280045,  0.000182644489,  0.001208139578],
        [0.101246280045,  0.000182644489,  0.001208139578],
        [0.101246280045,  0.000182644489,  0.001208139578]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "omega_RN_N Vector",
                                                               testFailCount, testMessages)

    #
    # check domega_RN_N
    #
    moduleOutput = dataLog.domega_RN_N
    # set the filtered output truth states
    trueVector = [
        [0.000000000000e+00,  -1.208139577635e-04,   1.826444892823e-05],
        [0.000000000000e+00,  -1.208139577635e-04,   1.826444892823e-05],
        [0.000000000000e+00,  -1.208139577635e-04,   1.826444892823e-05],
        [0.000000000000e+00,  -1.208139577635e-04,   1.826444892823e-05]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "domega_RN_N Vector",
                                                               testFailCount, testMessages)


    # If the argument provided at commandline "--show_plots" evaluates as true,
    # plot all figures
#    if show_plots:
#        # plot a sample variable.
#        plt.figure(1)
#        plt.plot(variableState[:,0]*macros.NANO2SEC, variableState[:,1], label='Sample Variable')
#        plt.legend(loc='upper left')
#        plt.xlabel('Time [s]')
#        plt.ylabel('Variable Description [unit]')
#        plt.show()

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]

def run2(show_plots):
    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Test times
    updateTime = 0.5  # update process rate update time
    totalTestSimTime = 1.5

    # Create test thread
    testProcessRate = mc.sec2nano(updateTime)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = eulerRotation.eulerRotation()
    module.ModelTag = "eulerRotation"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    angleSet = np.array([0.0, 90.0, 0.0]) * mc.D2R
    module.angleSet = angleSet
    angleRates = np.array([0.1, 0.0, 0.0]) * mc.D2R
    module.angleRates = angleRates

    # Create input message and size it because the regular creator of that message
    # is not part of the test.

    #
    # Reference Frame Message
    #
    RefStateOutData = messaging.AttRefMsgPayload()  # Create a structure for the input message
    sigma_R0N = np.array([0.1, 0.2, 0.3])
    RefStateOutData.sigma_RN = sigma_R0N
    omega_R0N_N = np.array([0.1, 0.0, 0.0])
    RefStateOutData.omega_RN_N = omega_R0N_N
    domega_R0N_N = np.array([0.0, 0.0, 0.0])
    RefStateOutData.domega_RN_N = domega_R0N_N
    attRefMsg = messaging.AttRefMsg().write(RefStateOutData)

    # Set the desired state and rate to 0.
    desiredAtt = messaging.AttStateMsgPayload()
    desiredState = np.array([0, 0, 0])
    desiredAtt.state = desiredState
    desiredRate = np.array([0, 0, 0])
    desiredAtt.rate = desiredRate
    desInMsg = messaging.AttStateMsg().write(desiredAtt)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.attRefInMsg.subscribeTo(attRefMsg)
    module.desiredAttInMsg.subscribeTo(desInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(mc.sec2nano(totalTestSimTime))  # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    accuracy = 1e-12
    #
    # check sigma_RN
    #
    moduleOutput = dataLog.sigma_RN
    # set the filtered output truth states
    trueVector = [
        [-0.193031238249, 0.608048400483, 0.386062476497],
        [-0.193031238249, 0.608048400483, 0.386062476497],
        [-0.193144351314, 0.607931107381, 0.386360300559],
        [-0.193257454832, 0.607813704445, 0.386658117585]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "sigma_RN Set",
                                                               testFailCount, testMessages)
    # print '\n sigma_RN = ', moduleOutput[:, 1:], '\n'
    #
    # check omega_RN_N
    #
    moduleOutput = dataLog.omega_RN_N
    # set the filtered output truth states
    trueVector = [
        [0.101246280045, 0.000182644489, 0.001208139578],
        [0.101246280045, 0.000182644489, 0.001208139578],
        [0.101246280045, 0.000182644489, 0.001208139578],
        [0.101246280045, 0.000182644489, 0.001208139578]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "omega_RN_N Vector",
                                                               testFailCount, testMessages)

    #
    # check domega_RN_N
    #
    moduleOutput = dataLog.domega_RN_N
    # set the filtered output truth states
    trueVector = [
        [0.000000000000e+00, -1.208139577635e-04, 1.826444892823e-05],
        [0.000000000000e+00, -1.208139577635e-04, 1.826444892823e-05],
        [0.000000000000e+00, -1.208139577635e-04, 1.826444892823e-05],
        [0.000000000000e+00, -1.208139577635e-04, 1.826444892823e-05]
    ]
    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "domega_RN_N Vector",
                                                               testFailCount, testMessages)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_all_test_eulerRotation(False)
    # run(False)
