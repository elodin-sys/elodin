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

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import smallBodyWaypointFeedback
from Basilisk.simulation import planetEphemeris
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport


# @pytest.mark.parametrize("accuracy", [1e-12])
# @pytest.mark.parametrize("param1, param2", [
#      (1, 1)
#     ,(1, 3)
# ])

def test_smallBodyWaypointFeedback(show_plots):
    r"""
    **Validation Test Description**

    This test checks two things: a large force output when the spacecraft is far from the waypoint, and a small force
    output when the spacecraft is at the waypoint.

    **Test Parameters**

    Args:
        :param show_plots: flag if plots should be shown.

    **Description of Variables Being Tested**

    In this test, the ``forceRequestBody`` variable in the :ref:`CmdForceBodyMsgPayload` output by the module is tested.
    When far away from the waypoint, the force request should be larger than 1 N. When close to the waypoint, the force
    request should only account for third body perturbations and SRP.
    """
    [testResults1, testMessages1] = smallBodyWaypointFeedbackTestFunction1()
    [testResults2, testMessages2] = smallBodyWaypointFeedbackTestFunction2()
    assert (testResults1 + testResults2) < 1, [testMessages1, testMessages2]


def smallBodyWaypointFeedbackTestFunction1():
    """This test checks for a large force return when far away from the waypoint"""
    testFailCount = 0
    testMessages = []

    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(0.5)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = smallBodyWaypointFeedback.SmallBodyWaypointFeedback()
    module.ModelTag = "smallBodyWaypointFeedback1"
    unitTestSim.AddModelToTask(unitTaskName, module)

    module.A_sc = 1.  # Surface area of the spacecraft, m^2
    module.M_sc = 300  # Mass of the spacecraft, kg
    module.IHubPntC_B = unitTestSupport.np2EigenMatrix3d([82.12, 0.0, 0.0, 0.0, 98.40, 0.0, 0.0, 0.0, 121.0])  # sc inertia
    module.mu_ast = 4.892  # Gravitational constant of the asteroid
    module.x1_ref = [-2000., 0., 0.]
    module.x2_ref = [0.0, 0.0, 0.0]
    module.K1 = unitTestSupport.np2EigenMatrix3d([5e-4, 0e-5, 0e-5, 0e-5, 5e-4, 0e-5, 0e-5, 0e-5, 5e-4])
    module.K2 = unitTestSupport.np2EigenMatrix3d([1., 0., 0., 0., 1., 0., 0., 0., 1.])

    # Set the orbital parameters of the asteroid
    oeAsteroid = planetEphemeris.ClassicElementsMsgPayload()
    oeAsteroid.a = 1.1259 * orbitalMotion.AU * 1000  # meters
    oeAsteroid.e = 0.20373
    oeAsteroid.i = 6.0343 * macros.D2R
    oeAsteroid.Omega = 2.01820 * macros.D2R
    oeAsteroid.omega = 66.304 * macros.D2R
    oeAsteroid.f = 346.32 * macros.D2R
    r_ON_N, v_ON_N = orbitalMotion.elem2rv(orbitalMotion.MU_SUN*(1000.**3), oeAsteroid)

    # Create the position and velocity of states of the s/c wrt the small body hill frame origin
    r_BO_N = np.array([-2000., 1500., 1000.]) # Position of the spacecraft relative to the body
    v_BO_N = np.array([0., 0., 0.])  # Velocity of the spacecraft relative to the body

    # Create the inertial position and velocity of the s/c
    r_BN_N = np.add(r_BO_N, r_ON_N)
    v_BN_N = np.add(v_BO_N, v_ON_N)

    # Configure blank module input messages
    asteroidEphemerisInMsgData = messaging.EphemerisMsgPayload()
    asteroidEphemerisInMsgData.r_BdyZero_N = r_ON_N
    asteroidEphemerisInMsgData.v_BdyZero_N = v_ON_N
    asteroidEphemerisInMsg = messaging.EphemerisMsg().write(asteroidEphemerisInMsgData)

    navTransInMsgData = messaging.NavTransMsgPayload()
    navTransInMsgData.r_BN_N = r_BN_N
    navTransInMsgData.v_BN_N = v_BN_N
    navTransInMsg = messaging.NavTransMsg().write(navTransInMsgData)

    navAttInMsgData = messaging.NavAttMsgPayload()
    navAttInMsgData.sigma_BN = np.array([0.1, 0.0, 0.0])
    navAttInMsg = messaging.NavAttMsg().write(navAttInMsgData)

    sunEphemerisInMsgData = messaging.EphemerisMsgPayload()
    sunEphemerisInMsg = messaging.EphemerisMsg().write(sunEphemerisInMsgData)

    # subscribe input messages to module
    module.navTransInMsg.subscribeTo(navTransInMsg)
    module.navAttInMsg.subscribeTo(navAttInMsg)
    module.asteroidEphemerisInMsg.subscribeTo(asteroidEphemerisInMsg)
    module.sunEphemerisInMsg.subscribeTo(sunEphemerisInMsg)

    # setup output message recorder objects
    forceOutMsgRec = module.forceOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, forceOutMsgRec)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.))
    unitTestSim.ExecuteSimulation()

    if np.linalg.norm(forceOutMsgRec.forceRequestBody) <= 1:
        testFailCount += 1
        testMessages.append("FAILED: " + module.ModelTag + " Module failed "
                            + "force output" + " unit test")

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]

def smallBodyWaypointFeedbackTestFunction2():
    """This test checks that the force output is near zero when at the waypoint"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(0.5)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = smallBodyWaypointFeedback.SmallBodyWaypointFeedback()
    module.ModelTag = "smallBodyWaypointFeedback2"
    unitTestSim.AddModelToTask(unitTaskName, module)

    module.A_sc = 1.  # Surface area of the spacecraft, m^2
    module.M_sc = 300  # Mass of the spacecraft, kg
    module.IHubPntC_B = unitTestSupport.np2EigenMatrix3d([82.12, 0.0, 0.0, 0.0, 98.40, 0.0, 0.0, 0.0, 121.0])  # sc inertia
    module.mu_ast = 4.892  # Gravitational constant of the asteroid
    module.x1_ref = [-2000., 0., 0.]
    module.x2_ref = [0.0, 0.0, 0.0]
    module.K1 = unitTestSupport.np2EigenMatrix3d([5e-4, 0e-5, 0e-5, 0e-5, 5e-4, 0e-5, 0e-5, 0e-5, 5e-4])
    module.K2 = unitTestSupport.np2EigenMatrix3d([1., 0., 0., 0., 1., 0., 0., 0., 1.])

    # Set the orbital parameters of the asteroid
    oeAsteroid = planetEphemeris.ClassicElementsMsgPayload()
    oeAsteroid.a = 1.1259 * orbitalMotion.AU * 1000  # meters
    oeAsteroid.e = 0.20373
    oeAsteroid.i = 6.0343 * macros.D2R
    oeAsteroid.Omega = 2.01820 * macros.D2R
    oeAsteroid.omega = 66.304 * macros.D2R
    oeAsteroid.f = 346.32 * macros.D2R
    r_ON_N, v_ON_N = orbitalMotion.elem2rv(orbitalMotion.MU_SUN*(1000.**3), oeAsteroid)

    # Create the position and velocity of states of the s/c wrt the small body hill frame
    r_BO_H = np.array([-2000., 0., 0.]) # Position of the spacecraft relative to the body
    v_BO_H = np.array([0., 0., 0.])  # Velocity of the spacecraft relative to the body

    r_BN_N, v_BN_N = orbitalMotion.hill2rv(r_ON_N, v_ON_N, r_BO_H, v_BO_H)

    # Configure blank module input messages
    asteroidEphemerisInMsgData = messaging.EphemerisMsgPayload()
    asteroidEphemerisInMsgData.r_BdyZero_N = r_ON_N
    asteroidEphemerisInMsgData.v_BdyZero_N = v_ON_N
    asteroidEphemerisInMsg = messaging.EphemerisMsg().write(asteroidEphemerisInMsgData)

    navTransInMsgData = messaging.NavTransMsgPayload()
    navTransInMsgData.r_BN_N = r_BN_N
    navTransInMsgData.v_BN_N = v_BN_N
    navTransInMsg = messaging.NavTransMsg().write(navTransInMsgData)

    navAttInMsgData = messaging.NavAttMsgPayload()
    navAttInMsgData.sigma_BN = np.array([0.1, 0.0, 0.0])
    navAttInMsg = messaging.NavAttMsg().write(navAttInMsgData)

    sunEphemerisInMsgData = messaging.EphemerisMsgPayload()
    sunEphemerisInMsg = messaging.EphemerisMsg().write(sunEphemerisInMsgData)

    # subscribe input messages to module
    module.navTransInMsg.subscribeTo(navTransInMsg)
    module.navAttInMsg.subscribeTo(navAttInMsg)
    module.asteroidEphemerisInMsg.subscribeTo(asteroidEphemerisInMsg)
    module.sunEphemerisInMsg.subscribeTo(sunEphemerisInMsg)

    # setup output message recorder objects
    forceOutMsgRec = module.forceOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, forceOutMsgRec)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.))
    unitTestSim.ExecuteSimulation()

    if np.linalg.norm(forceOutMsgRec.forceRequestBody) >= 1e-8:
        testFailCount += 1
        testMessages.append("FAILED: " + module.ModelTag + " Module failed "
                            + "force output" + " unit test")

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_smallBodyWaypointFeedback(False)


