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
from Basilisk.fswAlgorithms import smallBodyNavUKF
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros, orbitalMotion
from Basilisk.utilities import unitTestSupport
from matplotlib import pyplot as plt


def test_smallBodyNavUKF(show_plots):
    r"""
    **Validation Test Description**

    This unit test checks that the filter converges to a constant position and null velocity estimates under the presence of static measurements.
    Then, the non-Keplerian gravity estimation should match the Keplerian gravity with opposite sign.

    **Test Parameters**

    Args:
        :param show_plots: flag if plots should be shown.
    """
    [testResults, testMessage] = smallBodyNavUKFTestFunction(show_plots)
    assert testResults < 1, testMessage


def smallBodyNavUKFTestFunction(show_plots):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(15)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = smallBodyNavUKF.SmallBodyNavUKF()
    module.ModelTag = "smallBodyNavUKFTag"
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Set the filter parameters (hyperparameters, small body gravitational constant, noise matrices)
    module.alpha = 0  # Filter hyperparameter
    module.beta = 2  # Filter hyperparameter
    module.kappa = 1e-3  # Filter hyperparameter
    module.mu_ast = 17.2882449693*1e9  # Gravitational constant of the asteroid m^3/s^2
    module.P_proc = (0.1*np.identity(9)).tolist()  # Process Noise
    module.R_meas = (0.1*np.identity(3)).tolist()  # Measurement Noise

    vesta_radius = 2.3612 * orbitalMotion.AU * 1000  # meters
    vesta_velocity = np.sqrt(orbitalMotion.MU_SUN*(1000.**3)/vesta_radius) # m/s, assumes circular orbit

    x_0 = [2010., 1510., 1010., 0., 2., 0., 0.14, 0., 0.]
    module.x_hat_k = x_0
    module.P_k = [[1000., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 1000., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 1000., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1, 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 1, 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 1e-3, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 1e-3, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 1e-3]]
    #module.P_k = P_k.tolist()

    # Configure blank module input messages
    navTransInMsgData = messaging.NavTransMsgPayload()
    navTransInMsgData.r_BN_N = [vesta_radius + 600. * 1000., -400. * 1000, 200. * 1000]
    navTransInMsgData.v_BN_N = [0., vesta_velocity, 0.]
    navTransInMsg = messaging.NavTransMsg().write(navTransInMsgData)

    asteroidEphemerisInMsgData = messaging.EphemerisMsgPayload()
    asteroidEphemerisInMsgData.r_BdyZero_N = [vesta_radius, 0., 0.]
    asteroidEphemerisInMsgData.v_BdyZero_N = [0., vesta_velocity, 0.]
    asteroidEphemerisInMsgData.sigma_BN = [0.0, 0.0, 0.0]
    asteroidEphemerisInMsgData.omega_BN_B = [0.0, 0.0, 0.0]
    asteroidEphemerisInMsg = messaging.EphemerisMsg().write(asteroidEphemerisInMsgData)

    # subscribe input messages to module
    module.navTransInMsg.subscribeTo(navTransInMsg)
    module.asteroidEphemerisInMsg.subscribeTo(asteroidEphemerisInMsg)

    # setup output message recorder objects
    smallBodyNavUKFOutMsgRec = module.smallBodyNavUKFOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, smallBodyNavUKFOutMsgRec)
    smallBodyNavUKFOutMsgRecC = module.smallBodyNavUKFOutMsgC.recorder()
    unitTestSim.AddModelToTask(unitTaskName, smallBodyNavUKFOutMsgRecC)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(600))
    unitTestSim.ExecuteSimulation()

    x_hat = smallBodyNavUKFOutMsgRec.state
    x_hat_c_wrapped = smallBodyNavUKFOutMsgRecC.state
    covar = smallBodyNavUKFOutMsgRec.covar

    # Since the small body does not rotate, no inhomogeneous gravity has
    # been considered and the spacecraft velocity in the small body
    # fixed frame is null, then the measured acceleration should correspond
    # to the Keplerian gravity with opposite sign
    true_r = np.array([[600. * 1000, -400. * 1000, 200. * 1000]])
    true_v = np.array([[0., 0., 0.]])
    true_a = module.mu_ast * true_r / (np.linalg.norm(true_r))**3
    true_x_hat = np.zeros(9)
    true_x_hat[0:3] = true_r
    true_x_hat[3:6] = true_v
    true_x_hat[6:9] = true_a

    testFailCount, testMessages = unitTestSupport.compareArrayRelative(
        [true_x_hat], np.array([x_hat[-1,:]]), 0.01, "x_hat",
        testFailCount, testMessages)

    testFailCount, testMessages = unitTestSupport.compareArrayRelative(
        [true_x_hat], np.array([x_hat_c_wrapped[-1,:]]), 0.01, "x_hat_c_wrapped",
        testFailCount, testMessages)

    plt.figure(1)
    plt.clf()
    plt.figure(1, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.ticklabel_format(useOffset=False)
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,0] / 1000, label='x-pos')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,1] / 1000, label='y-pos')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,2] / 1000, label='z-pos')
    plt.legend(loc='lower left')
    plt.xlabel('Time (min)')
    plt.ylabel('${}^{A}r_{BA}$ (km)')
    plt.title('Estimated Relative Spacecraft Position')

    plt.figure(2)
    plt.clf()
    plt.figure(2, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,3], label='x-vel')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,4], label='y-vel')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,5], label='z-vel')
    plt.legend(loc='upper right')
    plt.xlabel('Time (min)')
    plt.ylabel('${}^{A}v_{BA}$ (m/s)')
    plt.title('Estimated Spacecraft Velocity')

    plt.figure(3)
    plt.clf()
    plt.figure(3, figsize=(7, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,6], label='x-acc')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,7], label='y-acc')
    plt.plot(smallBodyNavUKFOutMsgRec.times() * 1.0E-9 / 60, x_hat[:,8], label='z-acc')
    plt.legend(loc='lower right')
    plt.xlabel('Time (min)')
    plt.ylabel('${}^{A}a_{BA}$ (m/s^2)')
    plt.title('Estimated Non-Keplerian Acceleration')

    if show_plots:
        plt.show()

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_smallBodyNavUKF(True)
