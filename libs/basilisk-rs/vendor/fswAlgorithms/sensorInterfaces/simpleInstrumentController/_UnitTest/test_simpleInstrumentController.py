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
#   Module Name:        simpleInstrumentController
#   Author:             Adam Herrmann
#   Creation Date:      May 19th, 2020
#

import inspect
import os

import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)

# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.fswAlgorithms import simpleInstrumentController
from Basilisk.utilities import macros
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging

from matplotlib import pyplot as plt

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

tests = [(0, 0, 0, None, None, [1,0,0,1,0]), # rate disabled, device status not written, no controller status
         (0, 0.01, 0.1, None, None, [1,0,0,1,0]), # rate disabled, rate noncompliant, device status not written, no controller status
         (1, 0.01, 0.001, None, None, [1,0,0,1,0]), # rate enabled, rate compliant, device status not written, no controller status
         (1, 0.01, 0.1, None, None, [0, 0, 0, 0, 0]), # rate enabled, rate noncompliant, device status not written, no controller status
         (0, 0, 0, 1, 0, [1,0,0,1,0]), # rate disabled, device status of 1, controller status of 0
         (0, 0, 0, 0, 1, [0, 0, 0, 0, 0]), # rate disabled, device status of 0, controller status of 1
         (0, 0, 0, None, 1, [1,0,0,1,0]) # rate disabled, device status not written, controller status of 1
        ]
@pytest.mark.parametrize('use_rate_limit,rate_limit,omega_mag,deviceStatus,controlStatus,expected_result', tests)
def test_simple_instrument_controller(show_plots, use_rate_limit, rate_limit, omega_mag, deviceStatus, controlStatus, expected_result):
    r"""
    **Validation Test Description**

    Unit test for simpleInstrumentController. The unit test specifically covers:

    1. If the controller correctly sends an image command if access and attitude error are within bounds

    2. If the controller does not image until the imaged variable is reset to 0

    3. If the controller sends an image command again after the imaged variable has been reset to 0

    4. If the rate tolerance limit only effects operation when enabled, and effects operation in the expected manner.

    5. If the controller sends an image command when deviceStatusInMsg is set to 1 and the instrumentStatus is set to 0

    6. If the controller does not send an image command when deviceStatusInMsg is set to 0 and the instrumentStatus is set to 1

    7. If the controller does send an image command when deviceStatusInMsg is not written and the instrumentStatus is set to 1
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = simpleInstrumentControllerTestFunction(show_plots, use_rate_limit, rate_limit, omega_mag, deviceStatus, controlStatus, expected_result)
    
    assert testResults < 1, testMessage


def simpleInstrumentControllerTestFunction(show_plots, use_rate_limit=1, rate_limit=0.01, omega_mag=0.001, deviceStatus=None, controlStatus=None, expected_result=None):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(1.0)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = simpleInstrumentController.simpleInstrumentController()
    module.ModelTag = "simpleInstrumentController"           # update python name of test module

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.attErrTolerance = 0.1                           # set the attitude error tolerance
    module.rateErrTolerance = rate_limit                   # set the attitude rate error tolerance
    module.useRateTolerance = use_rate_limit               # enable attitude rate error tolerance

    # Create and write the ground location access message
    inputAccessMsgData = messaging.AccessMsgPayload()
    inputAccessMsgData.hasAccess = 1
    inputAccessMsg = messaging.AccessMsg().write(inputAccessMsgData)

    # Create and write the attitude guidance message
    inputAttGuidMsgData = messaging.AttGuidMsgPayload()
    inputAttGuidMsgData.sigma_BR = [0.01, 0.01, 0.01]
    inputAttGuidMsgData.omega_BR_B = [omega_mag, 0.0, 0.0]
    inputAttGuidMsg = messaging.AttGuidMsg().write(inputAttGuidMsgData)

    # Create and write the device status message
    if deviceStatus is not None:
        inputDeviceStatusMsgData = messaging.DeviceStatusMsgPayload()
        inputDeviceStatusMsgData.deviceStatus = deviceStatus
        inputDeviceStatusMsg = messaging.DeviceStatusMsg().write(inputDeviceStatusMsgData)
        module.deviceStatusInMsg.subscribeTo(inputDeviceStatusMsg)
    
    # Set the controllerStatus variable
    if controlStatus is not None:
        module.controllerStatus = controlStatus

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.deviceCmdOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect the message interfaces
    module.locationAccessInMsg.subscribeTo(inputAccessMsg)
    module.attGuidInMsg.subscribeTo(inputAttGuidMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # run the module again for an additional second
    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # Now change the imaged variable back to 0 and run again for another image
    module.imaged = 0
    unitTestSim.ConfigureStopTime(macros.sec2nano(4.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    if not unitTestSupport.isArrayEqual(dataLog.deviceCmd, expected_result, len(expected_result), 1e-12):
        testFailCount += 1
        testMessages.append("FAILED: " + module.ModelTag + " Module failed dataVector" + " unit test at t=" + str(dataLog.times()[0]*macros.NANO2SEC) + "sec\n")

    # Plots
    plt.close("all")  # close all prior figures so we start with a clean slate
    plt.figure(1)
    plt.plot(dataLog.times() * macros.NANO2SEC, dataLog.deviceCmd)
    plt.xlabel('Time [s]')
    plt.ylabel('Device Status')
    plt.suptitle('Device Status Over Time')

    if show_plots:
        plt.show()

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    simpleInstrumentControllerTestFunction(
        True        # show_plots
    )
