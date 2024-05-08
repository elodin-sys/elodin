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
#   Module Name:        ETcontrol
#   Author:             Julian Hammerl
#   Creation Date:      May 20, 2021
#

import numpy as np
import pytest
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import etSphericalControl  # import the module that is to be tested
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros, RigidBodyKinematics, orbitalMotion
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

@pytest.mark.parametrize("accuracy", [1e-8])
def test_etSphericalControl(show_plots, accuracy):     # update "module" in this function name to reflect the module name
    r"""
    **Validation Test Description**

    The behavior of the Electrostatic Tractor Spherical Relative Motion Control is tested. The electrostatic force
    between the servicer and the debris is calculated using a single sphere to represent each spacecraft. The simulation
    is run for a single update cycle and the resulting forces and torques acting on each body
    are compared to hand-computed truth values.

    **Test Parameters**

    Args:
        accuracy (float): relative accuracy value used in the validation tests

    **Description of Variables Being Tested**

    The module output messages for the inertial control force vector and body control force vector are compared to
    the truth values obtained from a Matlab simulation.
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = etSphericalControlTestFunction(show_plots, accuracy)
    assert testResults < 1, testMessage


def etSphericalControlTestFunction(show_plots, accuracy):
    """Test method"""
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    # Construct algorithm and associated C++ container
    module = etSphericalControl.etSphericalControl()
    module.ModelTag = "ETcontrol"           # update python name of test module

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    mu = 3.986004418e14  # [m^3/s^2] Earth's gravitational parameter

    L0 = 20.
    Ki = 4e-7
    Pi = 1.85 * Ki ** 0.5
    module.K = [Ki, 0.0, 0.0,
                      0.0, Ki, 0.0,
                      0.0, 0.0, Ki]
    module.P = [Pi, 0.0, 0.0,
                      0.0, Pi, 0.0,
                      0.0, 0.0, Pi]
    module.L_r = 30.
    module.theta_r = 0.
    module.phi_r = 0.
    module.mu = mu

    # Create input message and size it because the regular creator of that message
    # is not part of the test.

    oe = orbitalMotion.ClassicElements()
    oe.a = 42164. * 1e3  # [m] geostationary orbit
    oe.e = 0.
    oe.i = 10.*macros.D2R
    oe.Omega = 20.*macros.D2R
    oe.omega = 30.*macros.D2R
    oe.f = 40.*macros.D2R
    r_TN_N, v_TN_N = orbitalMotion.elem2rv(mu, oe)
    servicerNavTransOutData = messaging.NavTransMsgPayload()  # Create a structure for the input message
    servicerNavTransOutData.r_BN_N = r_TN_N
    servicerNavTransOutData.v_BN_N = v_TN_N
    servicerTransMsg = messaging.NavTransMsg().write(servicerNavTransOutData)

    r_DT_N = np.array([2., -L0, -3.])  # relative position between debris and servicer
    r_DN_N = r_TN_N + r_DT_N
    v_DN_N = v_TN_N
    debrisNavTransOutData = messaging.NavTransMsgPayload()  # Create a structure for the input message
    debrisNavTransOutData.r_BN_N = r_DN_N
    debrisNavTransOutData.v_BN_N = v_DN_N
    debrisTransMsg = messaging.NavTransMsg().write(debrisNavTransOutData)

    beta_TH = [0.972960339471760, 0.107600839071972, -0.0289291519077161, 0.202319898714648]  # initial EP
    sigma_TN = RigidBodyKinematics.EP2MRP(beta_TH)  # MRP
    servicerNavAttOutData = messaging.NavAttMsgPayload()
    servicerNavAttOutData.sigma_BN = sigma_TN
    servicerAttMsg = messaging.NavAttMsg().write(servicerNavAttOutData)

    servicerConfigOutData = messaging.VehicleConfigMsgPayload()
    servicerConfigOutData.massSC = 500.
    servicerVehicleConfigMsg = messaging.VehicleConfigMsg().write(servicerConfigOutData)

    debrisConfigOutData = messaging.VehicleConfigMsgPayload()
    debrisConfigOutData.massSC = 2000.
    debrisVehicleConfigMsg = messaging.VehicleConfigMsg().write(debrisConfigOutData)

    # compute electrostatic force between servicer and debris using single sphere for both S/C
    R_T = 2.
    R_D = 3.
    V_T = 25000.
    V_D = -25000.
    kc = 8.9875517923e9
    L = np.linalg.norm(r_DT_N)
    q_T = (L*(L*R_T*V_T-R_T*R_D*V_D))/(kc*(L**2.-R_T*R_D))
    q_D = (L * (L * R_D * V_D - R_T * R_D * V_T)) / (kc * (L ** 2. - R_T * R_D))
    Fc = kc*q_T*q_D/L**2.
    Fc_N = Fc*(-r_DT_N/np.linalg.norm(r_DT_N))  # electrostatic force acting on servicer
    eForceOutData = messaging.CmdForceInertialMsgPayload()
    eForceOutData.forceRequestInertial = Fc_N
    eForceMsg = messaging.CmdForceInertialMsg().write(eForceOutData)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLogInertial = module.forceInertialOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLogInertial)
    dataLogBody = module.forceBodyOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLogBody)

    # connect the message interfaces
    module.servicerTransInMsg.subscribeTo(servicerTransMsg)
    module.debrisTransInMsg.subscribeTo(debrisTransMsg)
    module.servicerAttInMsg.subscribeTo(servicerAttMsg)
    module.servicerVehicleConfigInMsg.subscribeTo(servicerVehicleConfigMsg)
    module.debrisVehicleConfigInMsg.subscribeTo(debrisVehicleConfigMsg)
    module.eForceInMsg.subscribeTo(eForceMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()
    unitTestSim.TotalSim.SingleStepProcesses()

    # This pulls the actual data log from the simulation run.
    forceInertialOutput = dataLogInertial.forceRequestInertial
    forceBodyOutput = dataLogBody.forceRequestBody

    # set the filtered output truth states
    trueInertialVector = [[-0.00714223893615245,
                           0.00267752848271998,
                           0.000883681113161883]]
    trueBodyVector = [[-0.00541988234898216,
                       0.00542736415350300,
                       0.000360862543207430]]

    # compare the module results to the truth values
    for i in range(0, len(trueInertialVector)):
        # check vector values
        if not unitTestSupport.isArrayEqual(forceInertialOutput[i], trueInertialVector[i], 3, accuracy):
            testFailCount += 1
            print(forceInertialOutput[i])
            testMessages.append("FAILED: " + module.ModelTag + " Module failed "
                                + "Inertial Force Output" + " unit test at t="
                                + str(forceInertialOutput[i, 0] * macros.NANO2SEC) + "sec\n")
        if not unitTestSupport.isArrayEqual(forceBodyOutput[i], trueBodyVector[i], 3, accuracy):
            testFailCount += 1
            print(forceBodyOutput[i])
            testMessages.append("FAILED: " + module.ModelTag + " Module failed "
                                + "Body Force Output" + " unit test at t="
                                + str(forceBodyOutput[i, 0] * macros.NANO2SEC) + "sec\n")

    #   print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
        print("This test uses an accuracy value of " + str(accuracy))
    else:
        print("FAILED " + module.ModelTag)
        print(testMessages)

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_etSphericalControl(
        False,        # show_plots
        1e-8       # accuracy
    )
