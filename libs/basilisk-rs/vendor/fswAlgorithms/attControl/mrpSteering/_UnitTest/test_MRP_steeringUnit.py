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

import matplotlib.pyplot as plt
import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import mrpSteering  # import the module that is to be tested
from Basilisk.utilities import RigidBodyKinematics
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


@pytest.mark.parametrize("K1", [0.15, 0])
@pytest.mark.parametrize("K3", [1, 0])
@pytest.mark.parametrize("omegaMax", [1.5 * macros.D2R, 0.001 * macros.D2R])


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
def test_mrp_steering_tracking(show_plots, K1, K3, omegaMax):
    r"""
    **Validation Test Description**

    This unit test  compares the computed :math:`\pmb\omega_{\mathcal{B}^{\ast}/\mathcal{R}}` and
    :math:`\pmb\omega_{\mathcal{B}^{\ast}/\mathcal{R}}'` to truth values computed in the python unit test.

    **Test Parameters**

    This test checks a set of gains ``K1``, ``K3`` and ``omegaMax`` on a rigid body with no external
    torques, and with a fixed input reference attitude message. The commanded rate solution
    is evaluated against python computed values at 0s, 0.5s and 1s to within a tolerance of :math:`10^{-12}`.

    :param show_plots: flag indicating if plots should be shown.
    :param K1: The control gain :math:`K_1`
    :param K3: The control gain :math:`K_3`
    :param omegaMax: The control gain :math:`\omega_{\text{max}}`
    :return: void

    """
    [testResults, testMessage] = mrp_steering_tracking(show_plots, K1, K3, omegaMax)
    assert testResults < 1, testMessage


def mrp_steering_tracking(show_plots, K1, K3, omegaMax):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
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
    module = mrpSteering.mrpSteering()
    module.ModelTag = "mrpSteering"


    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    module.K1 = K1
    module.K3 = K3
    module.omega_max = omegaMax

    #   Create input message and size it because the regular creator of that message
    #   is not part of the test.
    guidCmdData = messaging.AttGuidMsgPayload()  # Create a structure for the input message
    sigma_BR = np.array([0.3, -0.5, 0.7])
    guidCmdData.sigma_BR = sigma_BR
    omega_BR_B = np.array([0.010, -0.020, 0.015])
    guidCmdData.omega_BR_B = omega_BR_B
    omega_RN_B = np.array([-0.02, -0.01, 0.005])
    guidCmdData.omega_RN_B = omega_RN_B
    domega_RN_B = np.array([0.0002, 0.0003, 0.0001])
    guidCmdData.domega_RN_B = domega_RN_B
    guidInMsg = messaging.AttGuidMsg().write(guidCmdData)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.rateCmdOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.guidInMsg.subscribeTo(guidInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # Compute truth states
    omegaAstTrue, omegaAstPTrue = findTrueValues(guidCmdData, module)

    # compare the module results to the truth values
    accuracy = 1e-12
    for i in range(0, len(omegaAstTrue)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.omega_BastR_B[i], omegaAstTrue[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_BastR_B unit test at t="
                                + str(dataLog.times()[i] * macros.NANO2SEC) + "sec \n")

    # compare the module results to the truth values
    accuracy = 1e-12
    for i in range(0, len(omegaAstPTrue)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(dataLog.omegap_BastR_B[i], omegaAstPTrue[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omegap_BastR_B unit test at t="
                                + str(dataLog.times()[i] * macros.NANO2SEC) + "sec \n")

    # If the argument provided at commandline "--show_plots" evaluates as true,
    # plot all figures
    if show_plots:
        plt.show()

    # print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


def findTrueValues(guidCmdData, module):

    omegaMax = module.omega_max
    sigma = np.asarray(guidCmdData.sigma_BR)
    K1 = np.asarray(module.K1)
    K3 = np.asarray(module.K3)
    Bmat = RigidBodyKinematics.BmatMRP(sigma)
    omegaAst = []   #np.asarray([0, 0, 0])
    omegaAst_P = []

    for i in range(len(sigma)):
        steerRate = -1*(2*omegaMax/np.pi)*np.arctan((K1*sigma[i]+K3*sigma[i]*sigma[i]*sigma[i])*np.pi/(2*omegaMax))
        omegaAst.append(steerRate)


    if 1:   #module.ignoreOuterLoopFeedforward: #should be "if not"
        sigmaP = 0.25*Bmat.dot(omegaAst)
        for i in range(len(sigma)):
            omegaAstRate = (K1+3*K3*sigma[i]**2)/(1+((K1*sigma[i]+K3*sigma[i]**3)**2)*(np.pi/(2*omegaMax))**2)*sigmaP[i]
            omegaAst_P.append(-omegaAstRate)
    else:
        omegaAst_P = np.asarray([0, 0, 0])

    omegaAst = [omegaAst, omegaAst, omegaAst]
    omegaAst_P = [omegaAst_P, omegaAst_P, omegaAst_P]

    return omegaAst, omegaAst_P


if __name__ == "__main__":
    test_mrp_steering_tracking(False, 0.1, 1.0, 1.0)
