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
#   Module Name:        PRV_Steering
#   Author:             Hanspeter Schaub
#   Creation Date:      December 18, 2015
#
import matplotlib.pyplot as plt
# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.
import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import prvSteering
from Basilisk.fswAlgorithms import rateServoFullNonlinear
#   Import all of the modules that we are going to call in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_
@pytest.mark.parametrize("simCase", [0, 1])
def test_prvSteering(show_plots, simCase):     # update "subModule" in this function name to reflect the module name
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = subModuleTestFunction(show_plots, simCase)
    assert testResults < 1, testMessage


def subModuleTestFunction(show_plots, simCase):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    #   Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    #   Construct algorithm and associated C++ container
    module = prvSteering.prvSteering()
    module.ModelTag = "prvSteering"

    servo = rateServoFullNonlinear.rateServoFullNonlinear()
    servo.ModelTag = "rate_servo"

    #   Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)
    unitTestSim.AddModelToTask(unitTaskName, servo)

    # configure BSK modules
    module.K1 = 0.15
    module.K3 = 1.0
    module.omega_max = 1.5*macros.D2R
    servo.Ki = 0.01
    servo.P = 150.0
    servo.integralLimit = 2./servo.Ki * 0.1
    servo.knownTorquePntB_B = [0., 0., 0.]


    #   Create input message and size it because the regular creator of that message
    #   is not part of the test.

    #   attGuidOut Message:
    guidCmdData = messaging.AttGuidMsgPayload()  # Create a structure for the input message

    sigma_BR = []
    if simCase == 0:
        sigma_BR = np.array([0.3, -0.5, 0.7])
    if simCase == 1:
        sigma_BR = np.array([0, 0, 0])
    guidCmdData.sigma_BR = sigma_BR

    omega_BR_B = np.array([0.010, -0.020, 0.015])
    guidCmdData.omega_BR_B = omega_BR_B
    omega_RN_B = np.array([-0.02, -0.01, 0.005])
    guidCmdData.omega_RN_B = omega_RN_B
    domega_RN_B = np.array([0.0002, 0.0003, 0.0001])
    guidCmdData.domega_RN_B = domega_RN_B
    guidInMsg = messaging.AttGuidMsg().write(guidCmdData)

    # vehicleConfigData Message:
    vehicleConfigOut = messaging.VehicleConfigMsgPayload()
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    vehicleConfigOut.ISCPntB_B = I
    vcInMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

    # wheelSpeeds Message
    rwSpeedMessage = messaging.RWSpeedMsgPayload()
    Omega = [10.0, 25.0, 50.0, 100.0]
    rwSpeedMessage.wheelSpeeds = Omega
    rwSpeedInMsg = messaging.RWSpeedMsg().write(rwSpeedMessage)

    # wheelConfigData message
    def writeMsgInWheelConfiguration():
        rwConfigParams = messaging.RWArrayConfigMsgPayload()
        rwConfigParams.GsMatrix_B = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]
        rwConfigParams.JsList = [0.1, 0.1, 0.1, 0.1]
        rwConfigParams.numRW = 4
        rwParamInMsg = messaging.RWArrayConfigMsg().write(rwConfigParams)
        return rwParamInMsg

    rwParamInMsg = writeMsgInWheelConfiguration()

    # wheelAvailability message
    def writeMsgInWheelAvailability():
        rwAvailabilityMessage = messaging.RWAvailabilityMsgPayload()
        avail = [messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE]
        rwAvailabilityMessage.wheelAvailability = avail
        rwAvailInMsg = messaging.RWAvailabilityMsg().write(rwAvailabilityMessage)
        return rwAvailInMsg

    rwAvailInMsg = writeMsgInWheelAvailability()


    #   Setup logging on the test module output message so that we get all the writes to it
    dataLog = servo.cmdTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    #   connect input and output messages
    module.guidInMsg.subscribeTo(guidInMsg)
    servo.vehConfigInMsg.subscribeTo(vcInMsg)
    servo.guidInMsg.subscribeTo(guidInMsg)
    servo.rwParamsInMsg.subscribeTo(rwParamInMsg)
    servo.rwAvailInMsg.subscribeTo(rwAvailInMsg)
    servo.rwSpeedsInMsg.subscribeTo(rwSpeedInMsg)
    servo.rateSteeringInMsg.subscribeTo(module.rateCmdOutMsg)

    #   Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    #   Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    servo.Reset(1)     # this module reset function needs a time input (in NanoSeconds)

    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # set the filtered output truth states
    trueVector = []
    if simCase == 0:
        trueVector = [
                   [-2.9352922876097969, +6.2831737715827778, -4.0554726129822907]
                  ,[-2.9352922876097969, +6.2831737715827778, -4.0554726129822907]
                  ,[-2.9353853745179044, +6.2833455830962901, -4.0556481491012084]
                  ,[-2.9352922876097969, +6.2831737715827778, -4.0554726129822907]
                  ,[-2.9353853745179044, +6.2833455830962901, -4.0556481491012084]
                   ]
    if simCase == 1:
        trueVector = [
                     [-1.39,      3.79,     -1.39]
                    ,[-1.39,      3.79,     -1.39]
                    ,[-1.39005,   3.7901,   -1.390075]
                    ,[-1.39,      3.79,     -1.39]
                    ,[-1.39005,   3.7901,   -1.390075]
                     ]

    # compare the module results to the truth values
    accuracy = 1e-12
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.torqueRequestBody[i], trueVector[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed torqueRequestBody unit test at t="
                                + str(dataLog.times()[i]*macros.NANO2SEC) + "sec\n")







    # If the argument provided at commandline "--show_plots" evaluates as true,
    # plot all figures
    if show_plots:
          plt.show()

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]





#
#   This statement below ensures that the unitTestScript can be run as a stand-along python scripts
#   authmatically executes the runUnitTest() method
#
if __name__ == "__main__":
    test_prvSteering(True, 1)
