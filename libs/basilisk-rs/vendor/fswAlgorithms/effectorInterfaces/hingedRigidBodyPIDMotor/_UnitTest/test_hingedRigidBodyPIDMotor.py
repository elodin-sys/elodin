#
#  ISC License
#
#  Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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
#   Module Name:        solarArrayRotation
#   Author:             Riccardo Calaon
#   Creation Date:      January 12, 2023
#

import inspect
import os

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)



# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                   # general support file with common unit test functions
from Basilisk.fswAlgorithms import hingedRigidBodyPIDMotor       # import the module that is to be tested
from Basilisk.utilities import macros
from Basilisk.architecture import messaging                      # import the message definitions
from Basilisk.architecture import bskLogging


# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
# of the multiple test runs for this test.  Note that the order in that you add the parametrize method
# matters for the documentation in that it impacts the order in which the test arguments are shown.
# The first parametrize arguments are shown last in the pytest argument list
@pytest.mark.parametrize("thetaR", [np.pi/2, np.pi/6])
@pytest.mark.parametrize("thetaDotR", [0.1, 0.2])
@pytest.mark.parametrize("theta", [7*np.pi/6, 3*np.pi/2])
@pytest.mark.parametrize("thetaDot", [-0.1, -0.5])
@pytest.mark.parametrize("K", [0, 2])
@pytest.mark.parametrize("P", [0, 5])
@pytest.mark.parametrize("I", [0, 1])
@pytest.mark.parametrize("accuracy", [1e-8])


# update "module" in this function name to reflect the module name
def test_hingedRigidBodyPIDMotor(show_plots, thetaR, thetaDotR, theta, thetaDot, K, P, I, accuracy):
    r"""
    **Validation Test Description**

    This unit test verifies the correctness of the output motor torque :ref:`hingedRigidBodyPIDMotor`.
    The inputs provided are reference angle and angle rate, current angle and angle rate, proportional,
    derivative, and integral gain.

    **Test Parameters**

    Args:
        thetaR (double): reference angle;
        thetaDotR (double): reference angle rate;
        theta (double): current angle;
        thetaDot (double): current angle rate;
        K (double): proportional gain;
        P (double): derivative gain;
        I (double): integral gain;
        accuracy (float): absolute accuracy value used in the validation tests.

    **Description of Variables Being Tested**

    This unit test checks the correctness of the output motor torque

    - ``motorTorqueOutMsg``

    where in this case the output is one dimensional, since the spinning body is one dimensional. The unit test
    only checks the output at the first time step, for which the integral term does not contribute.
    """
    # each test method requires a single assert method to be called
    [testResults, testMessage] = hingedRigidBodyPIDMotorTestFunction(show_plots, thetaR, thetaDotR, theta, thetaDot, K, P, I, accuracy)
    assert testResults < 1, testMessage


def hingedRigidBodyPIDMotorTestFunction(show_plots, thetaR, thetaDotR, theta, thetaDot, K, P, I, accuracy):

    testFailCount = 0                        # zero unit test result counter
    testMessages = []                        # create empty array to store test log messages
    unitTaskName = "unitTask"                # arbitrary name (don't change)
    unitProcessName = "TestProcess"          # arbitrary name (don't change)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(1)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    motor = hingedRigidBodyPIDMotor.hingedRigidBodyPIDMotor()
    motor.ModelTag = "hingedRigidBodyPIDMotor"  
    motor.K = K
    motor.P = P
    motor.I = I
    unitTestSim.AddModelToTask(unitTaskName, motor)

    # Create input spinning body reference message
    refInMsgData = messaging.HingedRigidBodyMsgPayload()
    refInMsgData.theta = thetaR
    refInMsgData.thetaDot = thetaDotR
    refInMsg = messaging.HingedRigidBodyMsg().write(refInMsgData)
    motor.hingedRigidBodyRefInMsg.subscribeTo(refInMsg)

    # Create input spinning body message
    stateInMsgData = messaging.HingedRigidBodyMsgPayload()
    stateInMsgData.theta = theta
    stateInMsgData.thetaDot = thetaDot
    stateInMsg = messaging.HingedRigidBodyMsg().write(stateInMsgData)
    motor.hingedRigidBodyInMsg.subscribeTo(stateInMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = motor.motorTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.5))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    T = K * (thetaR - theta) + P * (thetaDotR - thetaDot)

    # compare the module results to the truth values
    if not unitTestSupport.isDoubleEqual(dataLog.motorTorque[0][0], T, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + motor.ModelTag + " module failed unit test for thetaR = {}, thetaDotR  = {}, theta = {}, thetaDot = {}, K = {}, P = {} \n".format(thetaR, thetaDotR, theta, thetaDot, K, P))

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_hingedRigidBodyPIDMotor( 
                 False,
                 np.pi/3,
                 0,
                 np.pi/2,
                 0.1,
                 1,
                 5,
                 0,
                 1e-12        # accuracy
               )
