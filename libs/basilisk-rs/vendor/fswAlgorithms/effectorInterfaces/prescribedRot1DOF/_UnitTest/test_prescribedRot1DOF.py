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


#
#   Unit Test Script
#   Module Name:        prescribedRot1DOF
#   Author:             Leah Kiner
#   Creation Date:      Nov 14, 2022
#

import pytest
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import prescribedRot1DOF  # import the module that is to be tested
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)


# Vary the initial angle, reference angle, and maximum angular acceleration for pytest
@pytest.mark.parametrize("thetaInit", [0, 2*np.pi/3])
@pytest.mark.parametrize("thetaRef", [0, 2*np.pi/3])
@pytest.mark.parametrize("thetaDDotMax", [0.008, 0.1])
@pytest.mark.parametrize("accuracy", [1e-12])
def test_prescribedRot1DOFTestFunction(show_plots, thetaInit, thetaRef, thetaDDotMax, accuracy):
    r"""
    **Validation Test Description**

    This unit test ensures that the profiled 1 DOF rotation for a secondary rigid body connected
    to the spacecraft hub is properly computed for a series of initial and reference PRV angles and maximum
    angular accelerations. The final prescribed attitude and angular velocity magnitude are compared with
    the reference values.

    **Test Parameters**

    Args:
        thetaInit (float): [rad] Initial PRV angle of the F frame with respect to the M frame
        thetaRef (float): [rad] Reference PRV angle of the F frame with respect to the M frame
        thetaDDotMax (float): [rad/s^2] Maximum angular acceleration for the attitude maneuver
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    This unit test ensures that the profiled 1 DOF rotation is properly computed for a series of initial and
    reference PRV angles and maximum angular accelerations. The final prescribed angle ``theta_FM_Final``
    and angular velocity magnitude ``thetaDot_Final`` are compared with the reference values ``theta_Ref`` and
    ``thetaDot_Ref``, respectively.
    """
    [testResults, testMessage] = prescribedRot1DOFTestFunction(show_plots, thetaInit, thetaRef, thetaDDotMax, accuracy)

    assert testResults < 1, testMessage


def prescribedRot1DOFTestFunction(show_plots, thetaInit, thetaRef, thetaDDotMax, accuracy):
    """Call this routine directly to run the unit test."""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create the test thread
    testProcessRate = macros.sec2nano(0.1)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Create an instance of the prescribedRot1DOF module to be tested
    PrescribedRot1DOF = prescribedRot1DOF.prescribedRot1DOF()
    PrescribedRot1DOF.ModelTag = "prescribedRot1DOF"

    # Add the prescribedRot1DOF test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, PrescribedRot1DOF)

    # Initialize the prescribedRot1DOF test module configuration data
    rotAxisM = np.array([1.0, 0.0, 0.0])
    prvInit_FM = thetaInit * rotAxisM
    PrescribedRot1DOF.rotAxis_M = rotAxisM
    PrescribedRot1DOF.thetaDDotMax = thetaDDotMax
    PrescribedRot1DOF.omega_FM_F = np.array([0.0, 0.0, 0.0])
    PrescribedRot1DOF.omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])
    PrescribedRot1DOF.sigma_FM = rbk.PRV2MRP(prvInit_FM)

    # Create the prescribedRot1DOF input message
    thetaDotRef = 0.0  # [rad/s]
    HingedRigidBodyMessageData = messaging.HingedRigidBodyMsgPayload()
    HingedRigidBodyMessageData.theta = thetaRef
    HingedRigidBodyMessageData.thetaDot = thetaDotRef
    HingedRigidBodyMessage = messaging.HingedRigidBodyMsg().write(HingedRigidBodyMessageData)
    PrescribedRot1DOF.spinningBodyInMsg.subscribeTo(HingedRigidBodyMessage)

    # Log the test module output message for data comparison
    dataLog = PrescribedRot1DOF.prescribedRotationOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    # Set the simulation time
    simTime = np.sqrt(((0.5 * np.abs(thetaRef - thetaInit)) * 8) / thetaDDotMax) + 1
    unitTestSim.ConfigureStopTime(macros.sec2nano(simTime))

    # Begin the simulation
    unitTestSim.ExecuteSimulation()

    # Extract the logged data for plotting and data comparison
    omega_FM_F = dataLog.omega_FM_F
    sigma_FM = dataLog.sigma_FM
    timespan = dataLog.times()

    thetaDot_Final = np.linalg.norm(omega_FM_F[-1, :])
    sigma_FM_Final = sigma_FM[-1, :]
    theta_FM_Final = 4 * np.arctan(np.linalg.norm(sigma_FM_Final))

    # Convert the logged sigma_FM MRPs to a scalar theta_FM array
    n = len(timespan)
    theta_FM = []
    for i in range(n):
        theta_FM.append(4 * np.arctan(np.linalg.norm(sigma_FM[i, :])))

    # Plot theta_FM
    thetaRef_plotting = np.ones(len(timespan)) * thetaRef
    thetaInit_plotting = np.ones(len(timespan)) * thetaInit
    plt.figure()
    plt.clf()
    plt.plot(timespan * macros.NANO2SEC, theta_FM, label=r"$\Phi$")
    plt.plot(timespan * macros.NANO2SEC, (180 / np.pi) * thetaRef_plotting, '--', label=r'$\Phi_{Ref}$')
    plt.plot(timespan * macros.NANO2SEC, (180 / np.pi) * thetaInit_plotting, '--', label=r'$\Phi_{0}$')
    plt.title(r'$\Phi_{\mathcal{F}/\mathcal{M}}$ Profiled Trajectory', fontsize=14)
    plt.ylabel('(deg)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='center right', prop={'size': 16})

    # Plot omega_FM_F
    plt.figure()
    plt.clf()
    plt.plot(timespan * macros.NANO2SEC, (180 / np.pi) * omega_FM_F[:, 0], label=r'$\omega_{1}$')
    plt.plot(timespan * macros.NANO2SEC, (180 / np.pi) * omega_FM_F[:, 1], label=r'$\omega_{2}$')
    plt.plot(timespan * macros.NANO2SEC, (180 / np.pi) * omega_FM_F[:, 2], label=r'$\omega_{3}$')
    plt.title(r'${}^\mathcal{F} \omega_{\mathcal{F}/\mathcal{M}}$ Profiled Trajectory', fontsize=14)
    plt.ylabel('(deg/s)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.legend(loc='upper right', prop={'size': 16})

    if show_plots:
        plt.show()
    plt.close("all")

    # Check to ensure the initial angle rate converged to the reference angle rate
    if not unitTestSupport.isDoubleEqual(thetaDot_Final, thetaDotRef, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + PrescribedRot1DOF.ModelTag + "thetaDot_Final and thetaDotRef do not match")

    # Check to ensure the initial angle converged to the reference angle
    if not unitTestSupport.isDoubleEqual(theta_FM_Final, thetaRef, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + PrescribedRot1DOF.ModelTag + "theta_FM_Final and thetaRef do not match")
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    prescribedRot1DOFTestFunction(
                 True,
                 np.pi/6,     # thetaInit
                 2*np.pi/3,     # thetaRef
                 0.008,       # thetaDDotMax
                 1e-12        # accuracy
               )
