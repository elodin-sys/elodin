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

import copy

import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import formationBarycenter
from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros, astroFunctions, orbitalMotion


@pytest.mark.parametrize("accuracy", [1e-8])


def test_formationBarycenter(show_plots, accuracy):
    r"""
    **Validation Test Description**

    This unit test verifies the formationBarycenter module. It checks the barycenter of three spacecraft using both
    cartesian coordinates and orbital elements weighted averaging.

    **Test Parameters**

    The test parameters used are the following:

    Args:
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    In this file we are checking the values of the variables

    - ``barycenter``
    - ``barycenterVelocity``
    - ``barycenterC``
    - ``barycenterVelocityC``

    which represent the center of mass position and velocity vectors. The variables ending in ``C`` are pulled from the
    C-wrapped navigation output message, whereas the other two come from the usual C++ message. All these variables are
    compared to ``trueBarycenter`` and ``trueBarycenterVelocity``, which contain their true values.

    As stated, both the C and C++ wrapped message outputs are checked.
    """
    [testResults, testMessage] = formationBarycenterTestFunction(show_plots, accuracy)
    assert testResults < 1, testMessage


def formationBarycenterTestFunction(show_plots, accuracy):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(1.)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    barycenterModule = formationBarycenter.FormationBarycenter()
    barycenterModule.ModelTag = "barycenterModuleTag"
    unitTestSim.AddModelToTask(unitTaskName, barycenterModule)

    # Configure each spacecraft's position and velocity
    mu = astroFunctions.mu_E

    oe1 = orbitalMotion.ClassicElements()
    oe1.a = 1.1 * astroFunctions.E_radius  # meters
    oe1.e = 0.01
    oe1.i = 45.0 * macros.D2R
    oe1.Omega = 48.2 * macros.D2R
    oe1.omega = 347.8 * macros.D2R
    oe1.f = 85.3 * macros.D2R
    rN1, vN1 = orbitalMotion.elem2rv(mu, oe1)

    oe2 = copy.deepcopy(oe1)
    oe2.e = 1.05 * oe1.e
    oe2.i = 1.05 * oe1.i
    oe2.f = 1.05 * oe1.f
    rN2, vN2 = orbitalMotion.elem2rv(mu, oe2)

    oe3 = copy.deepcopy(oe1)
    oe3.e = 0.95 * oe1.e
    oe3.i = 0.90 * oe1.i
    oe3.f = 1.10 * oe1.f
    rN3, vN3 = orbitalMotion.elem2rv(mu, oe3)

    # Configure spacecraft state input messages
    scNavMsgData1 = messaging.NavTransMsgPayload()
    scNavMsgData1.r_BN_N = rN1
    scNavMsgData1.v_BN_N = vN1
    scNavMsg1 = messaging.NavTransMsg().write(scNavMsgData1)

    scNavMsgData2 = messaging.NavTransMsgPayload()
    scNavMsgData2.r_BN_N = rN2
    scNavMsgData2.v_BN_N = vN2
    scNavMsg2 = messaging.NavTransMsg().write(scNavMsgData2)

    scNavMsgData3 = messaging.NavTransMsgPayload()
    scNavMsgData3.r_BN_N = rN3
    scNavMsgData3.v_BN_N = vN3
    scNavMsg3 = messaging.NavTransMsg().write(scNavMsgData3)

    # Configure spacecraft mass input messages
    scPayloadMsgData1 = messaging.VehicleConfigMsgPayload()
    scPayloadMsgData1.massSC = 100
    scPayloadMsg1 = messaging.VehicleConfigMsg().write(scPayloadMsgData1)

    scPayloadMsgData2 = messaging.VehicleConfigMsgPayload()
    scPayloadMsgData2.massSC = 150
    scPayloadMsg2 = messaging.VehicleConfigMsg().write(scPayloadMsgData2)

    scPayloadMsgData3 = messaging.VehicleConfigMsgPayload()
    scPayloadMsgData3.massSC = 250
    scPayloadMsg3 = messaging.VehicleConfigMsg().write(scPayloadMsgData3)

    # add spacecraft input messages to module
    barycenterModule.addSpacecraftToModel(scNavMsg1, scPayloadMsg1)
    barycenterModule.addSpacecraftToModel(scNavMsg2, scPayloadMsg2)
    barycenterModule.addSpacecraftToModel(scNavMsg3, scPayloadMsg3)

    # setup output message recorder objects
    barycenterOutMsg = barycenterModule.transOutMsg.recorder()
    barycenterOutMsgC = barycenterModule.transOutMsgC.recorder()
    unitTestSim.AddModelToTask(unitTaskName, barycenterOutMsg)
    unitTestSim.AddModelToTask(unitTaskName, barycenterOutMsgC)

    unitTestSim.InitializeSimulation()
    unitTestSim.TotalSim.SingleStepProcesses()

    barycenterModule.useOrbitalElements = True
    barycenterModule.mu = mu

    unitTestSim.TotalSim.SingleStepProcesses()

    # Pull module data
    barycenter = barycenterOutMsg.r_BN_N
    barycenterVelocity = barycenterOutMsg.v_BN_N
    barycenterC = barycenterOutMsgC.r_BN_N
    barycenterVelocityC = barycenterOutMsgC.v_BN_N
    elements = orbitalMotion.rv2elem(mu, barycenter[1], barycenterVelocity[1])
    elementsArray = [elements.a, elements.e, elements.i, elements.Omega, elements.omega, elements.f]
    elementsC = orbitalMotion.rv2elem(mu, barycenterC[1], barycenterVelocityC[1])
    elementsArrayC = [elementsC.a, elementsC.e, elementsC.i, elementsC.Omega, elementsC.omega, elementsC.f]

    # Set the true values
    trueBarycenter = np.array([-2795.61091086, 4349.07305245, 4711.56751498])
    trueBarycenterVelocity = np.array([-5.73871824, -4.74464078, 1.07961505])
    trueElements = [7015.94993, 0.0099, 0.7579092276785376, 0.8412486994612671, 6.07025513843626, 1.5855546253383273]

    # Verify the data
    if not unitTestSupport.isArrayEqual(barycenter[0], trueBarycenter, 3, accuracy) or \
            not unitTestSupport.isArrayEqual(barycenterVelocity[0], trueBarycenterVelocity, 3, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: formationBarycenter cartesian unit test.")

    if not unitTestSupport.isArrayEqual(elementsArray, trueElements, 6, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: formationBarycenter orbital element unit test.")

    if not unitTestSupport.isArrayEqual(barycenterC[0], trueBarycenter, 3, accuracy) or \
            not unitTestSupport.isArrayEqual(barycenterVelocityC[0], trueBarycenterVelocity, 3, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: formationBarycenter C message cartesian unit test.")

    if not unitTestSupport.isArrayEqual(elementsArrayC, trueElements, 6, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: formationBarycenter C message orbital element unit test.")

    if testFailCount == 0:
        print("PASSED: formationBarycenter unit test.")
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_formationBarycenter(
        False,  # show_plots
        1e-8  # accuracy
    )


