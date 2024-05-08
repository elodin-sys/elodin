# 
#  ISC License
# 
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder
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
#   Module Name:        locationPointing
#   Author:             Hanspeter Schaub
#   Creation Date:      May 9, 2021
#
import math

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import locationPointing
from Basilisk.utilities import RigidBodyKinematics
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport


@pytest.mark.parametrize("accuracy", [1e-12])
@pytest.mark.parametrize("r_LS_N", [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0, -1, 0.], [1, 1, 1]])
@pytest.mark.parametrize("locationType", [0, 1, 2])
@pytest.mark.parametrize("use3DRate", [True, False])
def test_locationPointing(show_plots, r_LS_N, locationType, use3DRate, accuracy):
    r"""
    **Validation Test Description**

    This unit test ensures that the Attitude Guidance and Attitude Reference messages content are properly computed
    for a series of desired inertial target locations
    

    **Test Parameters**

    Discuss the test parameters used.

    Args:
        r_LS_N (float): position vector of location relative to spacecraft
        locationType (int): choose whether to use ``locationInMsg``, ``celBodyInMsg`` or ``scTargetInMsg``
        use3DRate (bool): choose between 2D or 3D rate control
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    The script checks the attitude and rate outputs.

    """
    [testResults, testMessage] = locationPointingTestFunction(show_plots, r_LS_N, locationType,
                                                              use3DRate, accuracy)
    assert testResults < 1, testMessage


def locationPointingTestFunction(show_plots, r_LS_NIn, locationType, use3DRate, accuracy):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    timeStep = 0.1
    testProcessRate = macros.sec2nano(timeStep)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup
    pHat_B = np.array([1, 0, 0])
    r_SN_N = np.array([10, 11, 12])
    r_LS_N = np.array(r_LS_NIn)
    omega_BN_B = np.array([0.001, 0.002, 0.003])
    sigma_BN = np.array([0., 0., 0.])
    r_LN_N = r_LS_N + r_SN_N

    # setup module to be tested
    module = locationPointing.locationPointing()
    module.ModelTag = "locationPointingTag"
    unitTestSim.AddModelToTask(unitTaskName, module)
    module.pHat_B = pHat_B
    eps = 0.1 * macros.D2R
    module.smallAngle = eps
    if use3DRate:
        module.useBoresightRateDamping = 1

    # Configure input messages
    scTransInMsgData = messaging.NavTransMsgPayload()
    scTransInMsgData.r_BN_N = r_SN_N
    scTransInMsg = messaging.NavTransMsg().write(scTransInMsgData)
    scAttInMsgData = messaging.NavAttMsgPayload()
    scAttInMsgData.omega_BN_B = omega_BN_B
    scAttInMsgData.sigma_BN = sigma_BN
    scAttInMsg = messaging.NavAttMsg().write(scAttInMsgData)

    if locationType == 0:
        locationInMsgData = messaging.GroundStateMsgPayload()
        locationInMsgData.r_LN_N = r_LN_N
        locationInMsg = messaging.GroundStateMsg().write(locationInMsgData)
        module.locationInMsg.subscribeTo(locationInMsg)
    elif locationType == 1:
        locationInMsgData = messaging.EphemerisMsgPayload()
        locationInMsgData.r_BdyZero_N = r_LN_N
        locationInMsg = messaging.EphemerisMsg().write(locationInMsgData)
        module.celBodyInMsg.subscribeTo(locationInMsg)
    elif locationType == 2:
        locationInMsgData = messaging.NavTransMsgPayload()
        locationInMsgData.r_BN_N = r_LN_N
        locationInMsg = messaging.NavTransMsg().write(locationInMsgData)
        module.scTargetInMsg.subscribeTo(locationInMsg)

    # subscribe input messages to module
    module.scTransInMsg.subscribeTo(scTransInMsg)
    module.scAttInMsg.subscribeTo(scAttInMsg)

    # setup output message recorder objects
    attGuidOutMsgRec = module.attGuidOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, attGuidOutMsgRec)
    attRefOutMsgRec = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, attRefOutMsgRec)
    scTransRec = scTransInMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, scTransRec)
    scAttRec = scAttInMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, scAttRec)

    # setup and execute simulation
    unitTestSim.InitializeSimulation()
    counter = 0
    while counter < 3:
        scAttInMsgData.sigma_BN = sigma_BN + omega_BN_B * timeStep * counter
        scAttInMsg.write(scAttInMsgData)
        unitTestSim.ConfigureStopTime(macros.sec2nano(counter * timeStep))
        unitTestSim.ExecuteSimulation()
        counter += 1

    truthSigmaBR, truthOmegaBR, truthSigmaRN, truthOmegaRN = \
        truthValues(pHat_B, r_LN_N, r_SN_N, scAttRec.sigma_BN, scAttRec.omega_BN_B, eps, timeStep,
                    use3DRate)

    # compare the module results to the truth values
    for i in range(0, len(truthSigmaBR)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(attGuidOutMsgRec.sigma_BR[i], truthSigmaBR[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_BR unit test at t=" +
                                str(attGuidOutMsgRec.times()[i] * macros.NANO2SEC) +
                                "sec\n")

    for i in range(0, len(truthOmegaBR)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(attGuidOutMsgRec.omega_BR_B[i], truthOmegaBR[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_BR_B unit test at t=" +
                                str(attGuidOutMsgRec.times()[i] * macros.NANO2SEC) +
                                "sec\n")

    for i in range(0, len(truthSigmaRN)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(attRefOutMsgRec.sigma_RN[i], truthSigmaRN[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_RN unit test at t=" +
                                str(attRefOutMsgRec.times()[i] * macros.NANO2SEC) +
                                "sec\n")

    for i in range(0, len(truthOmegaRN)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(attRefOutMsgRec.omega_RN_N[i], truthOmegaRN[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed omega_RN_N unit test at t=" +
                                str(attRefOutMsgRec.times()[i] * macros.NANO2SEC) +
                                "sec\n")

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


def truthValues(pHat_B, r_LN_N, r_SN_N, sigma_BNList, omega_BNList, smallAngle, dt, use3DRate):
    # setup eHat180_B
    eHat180_B = np.cross(pHat_B, np.array([1., 0., 0.]))
    if np.linalg.norm(eHat180_B) < 0.1:
        eHat180_B = np.cross(pHat_B, np.array([0., 1., 0.]))
    eHat180_B = eHat180_B / np.linalg.norm(eHat180_B)

    r_LS_N = r_LN_N - r_SN_N

    counter = 0
    omega_BR_B = np.array([0., 0., 0.])
    sigma_BR_Out = []
    omega_BR_B_Out = []
    sigma_RN_Out = []
    omega_RN_N_Out = []
    while counter <= 2:
        sigma_BN = sigma_BNList[counter]
        dcmBN = RigidBodyKinematics.MRP2C(sigma_BN)
        r_LS_B = dcmBN.dot(r_LS_N)
        phi = math.acos(pHat_B.dot(r_LS_B) / np.linalg.norm(r_LS_B))
        if phi < smallAngle:
            sigma_BR = np.array([0., 0., 0.])
        else:
            if math.pi - phi < smallAngle:
                eHat_B = eHat180_B
            else:
                eHat_B = np.cross(pHat_B, r_LS_B)
            eHat_B = eHat_B / np.linalg.norm(eHat_B)
            sigma_BR = - math.tan(phi / 4.) * eHat_B

        if counter >= 1:
            dsigma = (sigma_BR - sigma_BR_Out[counter - 1]) / dt
            Binv = RigidBodyKinematics.BinvMRP(sigma_BR)
            omega_BR_B = Binv.dot(dsigma) * 4

        if use3DRate:
            rHat_LS_B = r_LS_B / np.linalg.norm(r_LS_B)
            omega_BR_B = omega_BR_B + (omega_BNList[counter].dot(rHat_LS_B))*rHat_LS_B

        # store truth results
        sigma_BR_Out.append(sigma_BR)
        omega_BR_B_Out.append(omega_BR_B)
        sigma_RN_Out.append(RigidBodyKinematics.addMRP(sigma_BNList[counter], -sigma_BR))
        omega_RN_N_Out.append(np.transpose(dcmBN).dot(omega_BNList[counter] - omega_BR_B_Out[counter]))

        counter += 1

    return sigma_BR_Out, omega_BR_B_Out, sigma_RN_Out, omega_RN_N_Out


if __name__ == "__main__":
    locationPointingTestFunction(False, [1, 0, 0], True, False, 1e-12)
