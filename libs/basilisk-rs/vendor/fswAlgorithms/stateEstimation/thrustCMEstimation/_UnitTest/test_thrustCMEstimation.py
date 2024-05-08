#
#  ISC License
#
#  Copyright (c) 2023 Laboratory for Atmospheric and Space Physics, University of Colorado at Boulder
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

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import thrustCMEstimation
from Basilisk.utilities import SimulationBaseClass, macros, unitTestSupport

bskPath = __path__[0]
fileName = os.path.basename(os.path.splitext(__file__)[0])


@pytest.mark.parametrize("dT", [10])  # s
@pytest.mark.parametrize("accuracy", [1e-12])
def test_thrustCMEstimation(show_plots, dT, accuracy):
    r"""
    **Validation Test Description**

    This unit test script tests the correctness of the center of mass (CM) estimate computed by
    :ref:`thrustCMEstimation` when multiple torque measurements are provided.

    **Test Parameters**

    This tests feeds several thrust vectors and thrust application points to the CM estimator module. These
    simulate the behavior of a gimbaled thruster.

    Args:
        dT (sec): time interval between two consecutive torque measurements are processed
        accuracy: tolerance on the result.

    **Description of Variables Being Tested**

    This Unit Test checks the correctness of the estimation based on the following considerations:
    - post-fit residuals are smaller than the associate pre-fit residuals for every measurement update;
    - post-fit residuals are within :math:`3\sigma` bounds of the measurement noise covariance;
    - the error on the state estimate is within :math:`3\sigma` bounds of the estimated covariance.
    """
    # each test method requires a single assert method to be called
    thrustCMEstimationTestFunction(show_plots, dT, accuracy)


def thrustCMEstimationTestFunction(show_plots, dT, accuracy):

    r_CB_B = np.array([0, 0, 0])       # exact CM location

    r_TB_B = np.array([[6, 5, 4],      # simulated thrust application point
                       [5, 4, 6],
                       [4, 6, 5],
                       [-6, 5, 4]])

    T_B = np.array([[1, 2, 3],         # simulated thrust vector
                    [2, 3, 1],
                    [3, 1, 2],
                    [1, -2, 3]])

    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(dT)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    #   setup the FSW algorithm tasks

    # setup thrustCMEstimation module
    cmEstimation = thrustCMEstimation.ThrustCMEstimation()
    cmEstimation.ModelTag = "cmEstimator"
    cmEstimation.attitudeTol = 1e-4
    cmEstimation.r_CB_B = [0.01, -0.025, 0.04]
    cmEstimation.P0 = [0.0025, 0.0025, 0.0025]
    cmEstimation.R0 = [1e-8, 1e-8, 1e-8]
    unitTestSim.AddModelToTask(unitTaskName, cmEstimation)

    # Write attitude guidance msg
    vehConfig = messaging.VehicleConfigMsgPayload()
    vehConfig.CoM = r_CB_B
    vehConfigMsg = messaging.VehicleConfigMsg().write(vehConfig)
    cmEstimation.vehConfigInMsg.subscribeTo(vehConfigMsg)

    # Write attitude guidance msg
    attGuidance = messaging.AttGuidMsgPayload()
    attGuidance.sigma_BR = [0, 0, 0]
    attGuidance.omega_BR_B = [0, 0, 0]
    attGuidMsg = messaging.AttGuidMsg().write(attGuidance)
    cmEstimation.attGuidInMsg.subscribeTo(attGuidMsg)

    # Write THR Config Msg in body frame coordinates B
    thrBConfig = messaging.THRConfigMsgPayload()
    thrConfigBMsg = messaging.THRConfigMsg()
    cmEstimation.thrusterConfigBInMsg.subscribeTo(thrConfigBMsg)

    # Write integral feedback torque
    intTorque = messaging.CmdTorqueBodyMsgPayload()
    intTorque.torqueRequestBody = [0, 0, 0]
    intFeedbackTorqueMsg = messaging.CmdTorqueBodyMsg()
    cmEstimation.intFeedbackTorqueInMsg.subscribeTo(intFeedbackTorqueMsg)

    #
    #   Setup data logging before the simulation is initialized
    #
    cmEstimateLog = cmEstimation.cmEstDataOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, cmEstimateLog)

    t = dT
    R0 = np.array([[cmEstimation.R0[0][0], 0, 0],
                   [0, cmEstimation.R0[1][0], 0],
                   [0, 0, cmEstimation.R0[2][0]]])
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(t-0.1))
    for i in range(len(r_TB_B)):
        thrBConfig.timeTag = macros.sec2nano(t)
        thrBConfig.rThrust_B = r_TB_B[i]
        thrBConfig.maxThrust = np.linalg.norm(T_B[i])
        thrBConfig.tHatThrust_B = T_B[i] / thrBConfig.maxThrust
        thrConfigBMsg.write(thrBConfig)

        intTorque.timeTag = macros.sec2nano(t)
        uMeasNoise = np.random.multivariate_normal([0,0,0], R0, size=1)
        intTorque.torqueRequestBody = -np.cross(r_TB_B[i], T_B[i]) + uMeasNoise[0]
        intFeedbackTorqueMsg.write(intTorque)

        t += dT
        unitTestSim.ExecuteSimulation()
        unitTestSim.ConfigureStopTime(macros.sec2nano(t-0.1))

    #   retrieve the logged data
    stateErr = cmEstimateLog.stateError
    sigma = cmEstimateLog.covariance
    preFit = cmEstimateLog.preFitRes
    postFit = cmEstimateLog.postFitRes
    
    # check that post-fit residuals are smaller in magnitude that pre-fit residuals at each measurement
    for i in range(len(r_TB_B)):
        np.testing.assert_array_less(np.linalg.norm(postFit[i]), np.linalg.norm(preFit[i]) + accuracy, verbose=True)

    # check that components of post-fit residuals are within 3-sigma bounds of measurement covariance
    for i in range(len(r_TB_B)):
        for j in range(3):
            np.testing.assert_array_less(postFit[i][j], 3*(R0[j][j])**0.5 + accuracy, verbose=True)

    # check that components of state errors are within 3-sigma bounds of state covariance
    for i in range(len(r_TB_B)):
        for j in range(3):
            np.testing.assert_array_less(stateErr[i][j], 3*sigma[i][j] + accuracy, verbose=True)

    return


#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_thrustCMEstimation(
        True,                # show_plots
        10,                  # dTsim
        1e-12                # accuracy
    )
