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
#   Module Name:        prescribedRot2DOF
#   Author:             Leah Kiner
#   Creation Date:      Nov 27, 2022
#

import pytest
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import prescribedRot2DOF  # import the module that is to be tested
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)

# Parametrize the user-configurable variables
@pytest.mark.parametrize("thetaInit", [0.01])
@pytest.mark.parametrize("thetaRef1a", [0.0, 2*np.pi/3])  # Rotation 1
@pytest.mark.parametrize("thetaRef2a", [np.pi/3, 2*np.pi/3])  # Rotation 1
@pytest.mark.parametrize("thetaRef1b", [0.0, 2*np.pi/3])  # Rotation 2
@pytest.mark.parametrize("thetaRef2b", [np.pi/3, 2*np.pi/3])  # Rotation 2
@pytest.mark.parametrize("phiDDotMax", [0.004])
@pytest.mark.parametrize("accuracy", [1e-5])
def test_PrescribedRot2DOFTestFunction(show_plots, thetaInit, thetaRef1a, thetaRef2a, thetaRef1b, thetaRef2b, phiDDotMax, accuracy):
    r"""
    **Validation Test Description**

    The unit test for this module simulates TWO consecutive 2 DOF rotations for a secondary rigid body connected
    to a rigid spacecraft hub. Two rotations are simulated to ensure that the module correctly updates
    the required relative PRV attitude when a new attitude reference message is written. This unit test checks that the
    prescribed body's MRP attitude converges to both reference attitudes for a series of initial and reference attitudes
    and maximum angular accelerations. (``sigma_FM_Final1`` is checked to converge to ``sigma_FM_Ref1``, and
    ``sigma_FM_Final2`` is checked to converge to ``sigma_FM_Ref2``). Additionally, the prescribed body's final angular
    velocity magnitude ``thetaDot_Final`` is checked for convergence to the reference angular velocity magnitude,
    ``thetaDot_Ref``.

    **Test Parameters**

    Args:
        thetaInit (float): [rad] Initial PRV angle of the F frame with respect to the M frame
        thetaRef1a (float): [rad] First reference PRV angle for the first rotation
        thetaRef2a (float): [rad] Second reference PRV angle for the first rotation
        thetaRef1b (float): [rad] First reference PRV angle for the second rotation
        thetaRef2b (float): [rad] Second reference PRV angle for the second rotation
        phiDDotMax (float): [rad/s^2] Maximum angular acceleration for the rotation
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    The prescribed body's MRP attitude at the end of the first rotation ``sigma_FM_Final1`` is checked to converge to
    the first reference attitude ``sigma_FM_Ref1``. The prescribed body's MRP attitude at the end of the second
    rotation ``sigma_FM_Final2`` is checked to converge to the second reference attitude ``sigma_FM_Ref2``.
    Additionally, the prescribed body's final angular velocity magnitude ``thetaDot_Final`` is checked for convergence
    to the reference angular velocity magnitude, ``thetaDot_Ref``.
    """

    [testResults, testMessage] = PrescribedRot2DOFTestFunction(show_plots, thetaInit, thetaRef1a, thetaRef2a, thetaRef1b, thetaRef2b, phiDDotMax, accuracy)

    assert testResults < 1, testMessage


def PrescribedRot2DOFTestFunction(show_plots, thetaInit, thetaRef1a, thetaRef2a, thetaRef1b, thetaRef2b, phiDDotMax, accuracy):
    """Call this routine directly to run the unit test."""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create the test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Create an instance of the =module that is being tested
    prescribedRot2DOFObj = prescribedRot2DOF.prescribedRot2DOF()
    prescribedRot2DOFObj.ModelTag = "PrescribedRot2DOF"

    # Initialize the test module configuration data
    rotAxis1_M = np.array([0.0, 1.0, 0.0])                                      # Rotation axis for the first reference rotation angle, thetaRef1a
    rotAxis2_F1 = np.array([0.0, 0.0, 1.0])                                     # Rotation axis for the second reference rotation angle, thetaRef2a
    prescribedRot2DOFObj.rotAxis1_M = rotAxis1_M
    prescribedRot2DOFObj.rotAxis2_F1 = rotAxis2_F1
    prescribedRot2DOFObj.phiDDotMax = phiDDotMax
    prescribedRot2DOFObj.omega_FM_F = np.array([0.0, 0.0, 0.0])              # [rad/s] Angular velocity of frame F relative to frame M in F frame components
    prescribedRot2DOFObj.omegaPrime_FM_F = np.array([0.0, 0.0, 0.0])         # [rad/s^2] B frame time derivative of omega_FB_F in F frame components
    prescribedRot2DOFObj.sigma_FM = np.array([0.0, 0.0, 0.0])                # MRP attitude of frame F relative to frame M

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, prescribedRot2DOFObj)

    # Create the prescribedRot2DOF input message
    thetaDot_Ref = 0.0  # [rad/s]
    hingedRigidBodyMessageData1 = messaging.HingedRigidBodyMsgPayload()
    hingedRigidBodyMessageData2 = messaging.HingedRigidBodyMsgPayload()
    hingedRigidBodyMessageData1.theta = thetaRef1a
    hingedRigidBodyMessageData2.theta = thetaRef2a
    hingedRigidBodyMessageData1.thetaDot = thetaDot_Ref
    hingedRigidBodyMessageData2.thetaDot = thetaDot_Ref
    HingedRigidBodyMessage1 = messaging.HingedRigidBodyMsg().write(hingedRigidBodyMessageData1)
    HingedRigidBodyMessage2 = messaging.HingedRigidBodyMsg().write(hingedRigidBodyMessageData2)
    prescribedRot2DOFObj.spinningBodyRef1InMsg.subscribeTo(HingedRigidBodyMessage1)
    prescribedRot2DOFObj.spinningBodyRef2InMsg.subscribeTo(HingedRigidBodyMessage2)

    # Set up message data recording logging on the test module output message to get access to it
    dataLog = prescribedRot2DOFObj.prescribedRotationOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Set up module variable data recording
    prescribedRot2DOFObjLog = prescribedRot2DOFObj.logger(["phi", "phiAccum"])
    unitTestSim.AddModelToTask(unitTaskName, prescribedRot2DOFObjLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    # Calculate the two reference PRVs for the first rotation
    prv_F0M_a = thetaRef1a * rotAxis1_M[0], thetaRef1a * rotAxis1_M[1], thetaRef1a * rotAxis1_M[2]
    prv_F1F0_a = thetaRef2a * rotAxis2_F1[0], thetaRef2a * rotAxis2_F1[1], thetaRef2a * rotAxis2_F1[2]

    # Calculate a single reference PRV for the first rotation and the associated MRP attitude
    if (thetaRef1a == 0 and thetaRef2a == 0):  # Prevent a (0,0,0) error using rbk.addPRV()
        prv_F1M_a = np.array([0.0, 0.0, 0.0])
        phi_F1M_a = 0.0
        sigma_FM_Ref1 = np.array([0.0, 0.0, 0.0])
    else:
        prv_F1M_a = rbk.addPRV(prv_F0M_a, prv_F1F0_a)
        phi_F1M_a = np.linalg.norm(prv_F1M_a)
        sigma_FM_Ref1 = rbk.PRV2MRP(prv_F1M_a)

    # Set the simulation time for the first rotation
    simTime1 = np.sqrt(((0.5 * np.abs(phi_F1M_a)) * 8) / phiDDotMax) + 10
    unitTestSim.ConfigureStopTime(macros.sec2nano(simTime1))

    # Execute the first rotation
    unitTestSim.ExecuteSimulation()

    # Extract the logged sigma_FM MRPs for data comparison
    sigma_FM_FirstMan = dataLog.sigma_FM
    sigma_FM_Final1 = sigma_FM_FirstMan[-1, :]

    # Calculate the two reference PRVs for the second rotation
    prv_F2M_b = thetaRef1b * rotAxis1_M[0], thetaRef1b * rotAxis1_M[1], thetaRef1b * rotAxis1_M[2]
    prv_F3F2_b = thetaRef2b * rotAxis2_F1[0], thetaRef2b * rotAxis2_F1[1], thetaRef2b * rotAxis2_F1[2]

    # Calculate a single reference PRV (prv_F3M_b) for the second rotation beginning from the M frame
    if (thetaRef1b == 0 and thetaRef2b == 0):  # Prevent a (0,0,0) error using rbk.addPRV()
        prv_F3M_b = np.array([0.0, 0.0, 0.0])
    else:
        prv_F3M_b = rbk.addPRV(prv_F2M_b, prv_F3F2_b)

    # Calculate a single reference PRV (prv_F3F1_b) for the second rotation beginning from the spinning body location after the first rotation (F1)
    # Also calculate the MRP representing the desired final attitude of the spinning body with respesct to the M frame
    if not unitTestSupport.isArrayEqual(prv_F1M_a, prv_F3M_b, 3, 1e-12):
        prv_F3F1_b = rbk.subPRV(prv_F1M_a, prv_F3M_b)
        sigma_FM_Ref2 = rbk.PRV2MRP(prv_F3M_b)
    else:
        prv_F3F1_b = np.array([0.0, 0.0, 0.0])
        sigma_FM_Ref2 = sigma_FM_Ref1
    phi_F3F1_b = np.linalg.norm(prv_F3F1_b)

    # Write the HingedRigidBody reference messages for the second rotation
    hingedRigidBodyMessageData1 = messaging.HingedRigidBodyMsgPayload()
    hingedRigidBodyMessageData2 = messaging.HingedRigidBodyMsgPayload()
    hingedRigidBodyMessageData1.theta = thetaRef1b
    hingedRigidBodyMessageData2.theta = thetaRef2b
    hingedRigidBodyMessageData1.thetaDot = thetaDot_Ref
    hingedRigidBodyMessageData2.thetaDot = thetaDot_Ref
    HingedRigidBodyMessage1 = messaging.HingedRigidBodyMsg().write(hingedRigidBodyMessageData1, macros.sec2nano(simTime1))
    HingedRigidBodyMessage2 = messaging.HingedRigidBodyMsg().write(hingedRigidBodyMessageData2, macros.sec2nano(simTime1))
    prescribedRot2DOFObj.spinningBodyRef1InMsg.subscribeTo(HingedRigidBodyMessage1)
    prescribedRot2DOFObj.spinningBodyRef2InMsg.subscribeTo(HingedRigidBodyMessage2)

    # Set the simulation time for the second rotation
    simTime2 = np.sqrt(((0.5 * np.abs(phi_F3F1_b)) * 8) / phiDDotMax) + 10
    unitTestSim.ConfigureStopTime(macros.sec2nano(simTime1 + simTime2))

    # Execute the second rotation
    unitTestSim.ExecuteSimulation()

    # Extract the recorded data for data comparison and plotting
    timespan = dataLog.times()
    omega_FM_F = dataLog.omega_FM_F
    sigma_FM = dataLog.sigma_FM

    # Extract the logged module variables
    phi = prescribedRot2DOFObjLog.phi
    phiAccum = prescribedRot2DOFObjLog.phiAccum

    # Store the final angular velocity of the spinning body
    thetaDot_Final = np.linalg.norm(omega_FM_F[-1, :])

    # Store the final MRP of the spinning body with respect to the M frame
    sigma_FM_Final2 = sigma_FM[-1, :]

    # Convert the logged omega_FM_F data to scalar thetaDot data
    n = len(timespan)
    thetaDot_FM = []
    for i in range(n):
        thetaDot_FM.append((np.linalg.norm(omega_FM_F[i, :])))

    # Plot omega_FB_F
    plt.figure()
    plt.clf()
    plt.plot(timespan * macros.NANO2SEC, omega_FM_F[:, 0], label=r'$\omega_{1}$')
    plt.plot(timespan * macros.NANO2SEC, omega_FM_F[:, 1], label=r'$\omega_{2}$')
    plt.plot(timespan * macros.NANO2SEC, omega_FM_F[:, 2], label=r'$\omega_{3}$')
    plt.title(r'Prescribed Angular Velocity ${}^\mathcal{F} \omega_{\mathcal{F}/\mathcal{M}}$')
    plt.xlabel('Time (s)')
    plt.ylabel('(rad/s)')
    plt.legend(loc='upper right', prop={'size': 12})

    # Plot phi
    thetaRef1_plotting = np.ones(len(timespan)) * phi_F1M_a
    thetaRef2_plotting = np.ones(len(timespan)) * phi_F3F1_b
    thetaInit_plotting = np.ones(len(timespan)) * thetaInit
    plt.figure()
    plt.clf()
    plt.plot(timespan * macros.NANO2SEC, phi, label=r'$\Phi$')
    plt.plot(timespan * macros.NANO2SEC, thetaInit_plotting, '--', label=r'$\Phi_{0}$')
    plt.plot(timespan * macros.NANO2SEC, thetaRef1_plotting, '--', label=r'$\Phi_{1_{Ref}}$')
    plt.plot(timespan * macros.NANO2SEC, thetaRef2_plotting, '--', label=r'$\Phi_{2_{Ref}}$')
    plt.title(r'Prescribed Principal Rotation Vector (PRV) Angles $\Phi$')
    plt.xlabel('Time (s)')
    plt.ylabel('(rad)')
    plt.legend(loc='upper right', prop={'size': 12})

    # Plot the accumulated PRV angle
    plt.figure()
    plt.clf()
    plt.plot(timespan * macros.NANO2SEC, phiAccum)
    plt.title(r'Accumulated Principal Rotation Vector (PRV) Angle $\Phi$')
    plt.xlabel('Time (s)')
    plt.ylabel('(rad)')

    if show_plots:
        plt.show()
    plt.close("all")

    # Compare the reference and simulated data and output failure messages as necessary
    if not unitTestSupport.isDoubleEqual(thetaDot_Final, thetaDot_Ref, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + prescribedRot2DOFObj.ModelTag + " thetaDot_Final and thetaDot_Ref do not match")
        print("thetaDot_Final: ")
        print(thetaDot_Final)
        print("thetaDot_Ref: ")
        print(thetaDot_Ref)

    if not unitTestSupport.isArrayEqual(sigma_FM_Final1, sigma_FM_Ref1, 3, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + prescribedRot2DOFObj.ModelTag + " MRPs sigma_FM_Final1 and sigma_FM_Ref1 do not match")
        print("sigma_FM_Final1: ")
        print(sigma_FM_Final1)
        print("sigma_FM_Ref1: ")
        print(sigma_FM_Ref1)

    if not unitTestSupport.isArrayEqual(sigma_FM_Final2, sigma_FM_Ref2, 3, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + prescribedRot2DOFObj.ModelTag + " MRPs sigma_FM_Final2 and sigma_FM_Ref2 do not match")
        print("sigma_FM_Final2: ")
        print(sigma_FM_Final2)
        print("sigma_FM_Ref2: ")
        print(sigma_FM_Ref2)

    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    PrescribedRot2DOFTestFunction(
                 True,
                 0.0,              # thetaInit
                 2 * np.pi / 3,    # thetaRef1a
                 np.pi / 6,        # thetaRef2a
                 0.0,              # thetaRef1b
                 2 * np.pi / 3,    # thetaRef2b
                 0.008,            # phiDDotMax
                 1e-5              # accuracy
               )
