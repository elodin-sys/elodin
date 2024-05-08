#
#  ISC License
#
#  Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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
import os, inspect, random
import numpy as np

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)


# Import all the modules that are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.fswAlgorithms import thrusterPlatformState
from Basilisk.utilities import macros
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.architecture import messaging
from Basilisk.architecture import bskLogging


@pytest.mark.parametrize("theta1", [0, np.pi/36, np.pi/18])
@pytest.mark.parametrize("theta2", [0, np.pi/36, np.pi/18])
@pytest.mark.parametrize("accuracy", [1e-10])
# update "module" in this function name to reflect the module name
def test_platformRotation(show_plots, theta1, theta2, accuracy):
    r"""
    **Validation Test Description**

    This unit test script tests the correctness of the output thruster configuration msg outputted by
    :ref:`thrusterPlatformState`. The correctness of the output is determined checking that the thrust unit direction
    vector, magnitude, and application point, match the rigid body rotation described by the input tip and tild angles
    theta1 and theta2.

    **Test Parameters**

    This test provides input tip and tilt angles to the module, as well as the thruster configuration information
    expressed with respect to the platform frame F.

    Args:
        theta1 (rad): platform tip angle
        theta2 (rad): platform tilt angle
        accuracy (float): accuracy within which results are considered to match the truth values.

    **Description of Variables Being Tested**

    In this test, offsets are given between the thrust application point and the origin of the platform frame
    (:math:`r_{T/F}`), and between the origin of the platform frame and the origin of the mount frame (:math:`r_{F/M}`).
    These offset vectors are hard coded into the unit test. The test checks the correctness of the output thrust unit
    direction vector and magnitude in the body frame, as well as the thrust application point location with respect to
    the origin of the body frame B, in body frame coordinates.
    """
    # each test method requires a single assert method to be called
    platformRotationTestFunction(show_plots, theta1, theta2, accuracy)


def platformRotationTestFunction(show_plots, theta1, theta2, accuracy):

    sigma_MB = np.array([0., 0., 0.])
    r_BM_M = np.array([0.0, 0.1, 1.4])
    r_FM_F = np.array([0.0, 0.0, -0.1])
    r_TF_F = np.array([-0.01, 0.03, 0.02])
    T_F    = np.array([1.0, 1.0, 10.0])

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
    platform = thrusterPlatformState.thrusterPlatformState()
    platform.ModelTag = "platformReference"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, platform)

    # Initialize the test module configuration data
    platform.sigma_MB = sigma_MB
    platform.r_BM_M = r_BM_M
    platform.r_FM_F = r_FM_F

    # Create input THR Config Msg
    THRConfig = messaging.THRConfigMsgPayload()
    THRConfig.rThrust_B = r_TF_F
    THRConfig.maxThrust = np.linalg.norm(T_F)
    THRConfig.tHatThrust_B = T_F / THRConfig.maxThrust
    thrConfigFMsg = messaging.THRConfigMsg().write(THRConfig)
    platform.thrusterConfigFInMsg.subscribeTo(thrConfigFMsg)

    # Create input hinged rigid body messages
    hingedBodyMsg1 = messaging.HingedRigidBodyMsgPayload()
    hingedBodyMsg1.theta = theta1
    hingedBody1InMsg = messaging.HingedRigidBodyMsg().write(hingedBodyMsg1)
    platform.hingedRigidBody1InMsg.subscribeTo(hingedBody1InMsg)
    hingedBodyMsg2 = messaging.HingedRigidBodyMsgPayload()
    hingedBodyMsg2.theta = theta2
    hingedBody2InMsg = messaging.HingedRigidBodyMsg().write(hingedBodyMsg2)
    platform.hingedRigidBody2InMsg.subscribeTo(hingedBody2InMsg)

    # Setup logging on the test module output messages so that we get all the writes to it
    thrConfigLog = platform.thrusterConfigBOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, thrConfigLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.5))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    rThrust_B = thrConfigLog.rThrust_B[0]
    tHatThrust_B = thrConfigLog.tHatThrust_B[0]
    tMax = thrConfigLog.maxThrust[0]

    FM = rbk.euler1232C([theta1, theta2, 0.0])
    MB = rbk.MRP2C(sigma_MB)
    FB = np.matmul(FM, MB)

    r_TB_B = np.matmul(FB.transpose(), r_TF_F + r_FM_F - np.matmul(FM, r_BM_M))     # thrust application point
    tHat_B = np.matmul(FB.transpose(), T_F) / np.linalg.norm(T_F)                   # thrust unit direction vector

    np.testing.assert_allclose(rThrust_B, r_TB_B, rtol=0, atol=accuracy, verbose=True)
    np.testing.assert_allclose(tHatThrust_B, tHat_B, rtol=0, atol=accuracy, verbose=True)
    np.testing.assert_allclose(tMax, np.linalg.norm(T_F), rtol=0, atol=accuracy, verbose=True)

    return


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_platformRotation(
                 False,                   # show_plots
                 0,                       # theta1
                 np.pi/36,                # theta2
                 1e-10                    # accuracy
    )
