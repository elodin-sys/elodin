# 
#  ISC License
# 
#  Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado Boulder
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

import pytest
import os
import inspect
import numpy as np

from Basilisk.utilities import SimulationBaseClass
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
from Basilisk.fswAlgorithms import scanningInstrumentController
from Basilisk.architecture import bskLogging

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)

#Test cases:
#1: attitude compliant, rate disabled, device status written, controller status of 1
#2: attitude noncompliant, rate disabled, device status written, controller status of 1
#3: attitude compliant, rate disabled, device status not written, no controller status
#4: attitude compliant, rate disabled, device status of 1, controller status of 0
#5: attitude compliant, rate disabled, device status of 0, controller status of 1
#6: attitude compliant, rate disabled, device status not written, controller status of 1
#7: attitude compliant, rate disabled, rate noncompliant, device status not written, no 
# controller status
#8: attitude compliant, rate enabled, rate noncompliant, device status not written, no 
# controller status
#9: attitude compliant, rate enabled, rate compliant, device status not written, no 
# controller status
#10: attitude noncompliant, rate enabled, rate compliant
tests = [(0.1, 0.01, 0, 0, 0, 1, 1, [1, 1, 1]), 
         (0.1, 0.2, 0, 0, 0, 1, 1, [0, 0, 0]), 
         (0.1, 0.01, 0, 0, 0, None, None, [0, 0, 0]), 
         (0.1, 0.01, 0, 0, 0, 1, 0, [1, 1, 1]), 
         (0.1, 0.01, 0, 0, 0, 0, 1, [0, 0, 0]), 
         (0.1, 0.01, 0, 0, 0, None, 1, [1, 1, 1]), 
         (0.1, 0.01, 0, 0.01, 0.1, None, 1, [1, 1, 1]), 
         (0.1, 0.01, 1, 0.01, 0.1, None, 1, [0, 0, 0]), 
         (0.1, 0.01, 1, 0.01, 0.001, 1, None, [1, 1, 1]), 
         (0.1, 0.2, 1, 0.01, 0.001, 1, None, [0, 0, 0]) 
        ]

@pytest.mark.parametrize('att_limit, att_mag, use_rate_limit,rate_limit,omega_mag' + 
                         ',deviceStatus,controlStatus,expected_result', tests)
def test_scanningInstrumentController(att_limit, att_mag, use_rate_limit, rate_limit, 
                                    omega_mag, deviceStatus, controlStatus, 
                                    expected_result):
    r"""
    **Validation Test Description**

    This test verifies if the module is working properly checking for input conditions
    such as rate limit, attitude error, and device status. 

    **Test Parameters**

    Args:
        att_limit (float): The attitude error tolerance
        att_mag (float): The magnitude of the attitude
        use_rate_limit (int): Determines if the rate limit requirement is enabled
        rate_limit (float): The rate limit requirement
        omega_mag (float): The magnitude of the angular velocity
        deviceStatus (int): The instrument status (on/off)
        controlStatus (int): The controller status (on/off)
        expected_result (list): The expected output of the module

    **Description of Variables Being Tested**

    The test checks the data stored in deviceCmdOutMsg and compares it to the expected 
    result. 
    """
    
    module_results = scanningInstrumentControllerTestFunction(att_limit, 
                                        att_mag, use_rate_limit, rate_limit, omega_mag, 
                                        deviceStatus, controlStatus, expected_result)

    np.testing.assert_array_equal(module_results, expected_result)


def scanningInstrumentControllerTestFunction(att_limit = 0.1, att_mag = 0.1, 
                        use_rate_limit=1, rate_limit=0.01, omega_mag=0.001, 
                        deviceStatus=None, controlStatus=None, expected_result=None):
    """Test method"""
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(1.0)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = scanningInstrumentController.scanningInstrumentController()
    module.ModelTag = "scanningInstrumentControllerTag"
    unitTestSim.AddModelToTask(unitTaskName, module)

    #Initializing the test module configuration data
    module.useRateTolerance = use_rate_limit              # rate limit enabled
    module.rateErrTolerance = rate_limit                  # rate limit
    module.attErrTolerance = att_limit                    # attitude error tolerance

    # Configure blank module input messages
    accessInMsgData = messaging.AccessMsgPayload()
    accessInMsgData.hasAccess = 1                 # set access to true
    accessInMsg = messaging.AccessMsg().write(accessInMsgData)

    attGuidInMsgData = messaging.AttGuidMsgPayload()
    attGuidInMsgData.sigma_BR = [att_mag, 0, 0]
    attGuidInMsgData.omega_BR_B = [omega_mag, 0.0, 0.0]
    attGuidInMsg = messaging.AttGuidMsg().write(attGuidInMsgData)

    if deviceStatus is not None:
        deviceStatusInMsgData = messaging.DeviceStatusMsgPayload()
        deviceStatusInMsgData.deviceStatus = deviceStatus
        deviceStatusInMsg = messaging.DeviceStatusMsg().write(deviceStatusInMsgData)
        module.deviceStatusInMsg.subscribeTo(deviceStatusInMsg)

    if controlStatus is not None:
        module.controllerStatus = controlStatus

    # subscribe input messages to module
    module.accessInMsg.subscribeTo(accessInMsg)
    module.attGuidInMsg.subscribeTo(attGuidInMsg)

    # setup output message recorder objects
    deviceCmdOutMsgRec = module.deviceCmdOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, deviceCmdOutMsgRec)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))
    unitTestSim.ExecuteSimulation()

    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))
    unitTestSim.ExecuteSimulation()

    return deviceCmdOutMsgRec.deviceCmd


if __name__ == "__main__":
    test_scanningInstrumentController(0.1, 0.01, 0, 0, 0, 1, 1, [1, 1, 1])