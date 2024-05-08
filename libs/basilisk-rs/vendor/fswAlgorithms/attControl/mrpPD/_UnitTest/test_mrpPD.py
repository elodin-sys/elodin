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
import inspect
import os

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))







from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions
from Basilisk.fswAlgorithms import mrpPD  # import the module that is to be tested
from Basilisk.utilities import macros
from Basilisk.architecture import messaging

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_

@pytest.mark.parametrize("setExtTorque", [False, True])

def test_mrp_PD_tracking(show_plots, setExtTorque):
    r"""
    **Validation Test Description**

    The unit test  for this module is kept as there are no branching code segments to account for different cases.
    The spacecraft inertia tensor message is setup, as well as a guidance message.  The module is then run for a
    few time steps and the control torque output message compared to a known answer.  The simulation only variable
    is if the known external torque is specified, or if the zero default vector is used.

    **Test Parameters**

    The unit test verifies that the module output torque message vector matches expected values.  The test
    method parameters include the following.

    :param show_plots: flag to show the test run plots
    :param setExtTorque: flag to set the knownTorquePntB_B variable
    :return: void

    """
    [testResults, testMessage] = mrp_PD_tracking(show_plots, setExtTorque)
    assert testResults < 1, testMessage


def mrp_PD_tracking(show_plots, setExtTorque):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_PD_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    module = mrpPD.mrpPD()
    module.ModelTag = "mrpPD"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.K = 0.15
    module.P = 150.0
    if setExtTorque:
        module.knownTorquePntB_B = [0.1, 0.2, 0.3]

    #   Create input message and size it because the regular creator of that message
    #   is not part of the test.
    #   attGuidOut Message:
    guidCmdData = messaging.AttGuidMsgPayload()
    guidCmdData.sigma_BR = [0.3, -0.5, 0.7]
    guidCmdData.omega_BR_B = [0.010, -0.020, 0.015]
    guidCmdData.omega_RN_B = [-0.02, -0.01, 0.005]
    guidCmdData.domega_RN_B = [0.0002, 0.0003, 0.0001]
    guidInMsg = messaging.AttGuidMsg().write(guidCmdData)

    # vehicleConfig FSW Message:
    vehicleConfigIn = messaging.VehicleConfigMsgPayload()
    vehicleConfigIn.ISCPntB_B = [1000., 0., 0.,
                                  0., 800., 0.,
                                  0., 0., 800.]
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigIn)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.cmdTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.vehConfigInMsg.subscribeTo(vcInMsg)
    module.guidInMsg.subscribeTo(guidInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    trueVector = [findTrueTorques(module, guidCmdData, vehicleConfigIn)]*3
    # print trueVector

    # compare the module results to the truth values
    accuracy = 1e-12
    print("accuracy = " + str(accuracy))

    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, dataLog.torqueRequestBody, accuracy,
                                                               "torqueRequestBody", testFailCount, testMessages)

    snippentName = "passFail" + str(setExtTorque)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


def findTrueTorques(module, guidCmdData, vehicleConfigOut):
    sigma_BR = np.array(guidCmdData.sigma_BR)
    omega_BR_B = np.array(guidCmdData.omega_BR_B)
    omega_RN_B = np.array(guidCmdData.omega_RN_B)
    domega_RN_B = np.array(guidCmdData.domega_RN_B)

    I = np.identity(3)
    I[0][0] = vehicleConfigOut.ISCPntB_B[0]
    I[1][1] = vehicleConfigOut.ISCPntB_B[4]
    I[2][2] = vehicleConfigOut.ISCPntB_B[8]

    K = module.K
    P = module.P
    L = np.array(module.knownTorquePntB_B)

    # Begin Method
    omega_BN_B = omega_BR_B + omega_RN_B
    temp1 = np.dot(I, omega_BN_B)
    temp2 = domega_RN_B - np.cross(omega_BN_B, omega_RN_B)
    Lr = K * sigma_BR + P * omega_BR_B - np.cross(omega_RN_B, temp1) - np.dot(I, temp2)
    Lr += L
    Lr *= -1.0

    return Lr



if __name__ == "__main__":
    test_mrp_PD_tracking(False, False)
