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
#   Module Name:        waypointReference
#   Author:             Riccardo Calaon
#   Creation Date:      March 14, 2021
#


import os

import numpy as np
import pytest
from Basilisk.architecture import bskLogging
from Basilisk.fswAlgorithms import waypointReference
from Basilisk.utilities import RigidBodyKinematics as rbk
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

path = os.path.dirname(os.path.abspath(__file__))
dataFileName = None
# The following 'parametrize' function decorator provides the parameters and expected results for each
# of the multiple test runs for this test.
@pytest.mark.parametrize("attType", [0, 1, 2])
@pytest.mark.parametrize("MRPswitching", [True, False])
@pytest.mark.parametrize("useReferenceFrame", [True, False])
@pytest.mark.parametrize("accuracy", [1e-12])

# provide a unique test method name, starting with test_
def test_waypointReference(show_plots, attType, MRPswitching, useReferenceFrame, accuracy):
    r"""
    **Validation Test Description**

    This unit test script tests the capability of the WaypointReference module to correctly read time-tagged
    attitude parameters, angular rates and angular accelerations from a text file. 
    First a text file is generated that contains a sequence of time-tagged attitude parameters, angular rates 
    and angular accelerations; subsequently, the same file is fed to the waypointReference module.
    The module is tested against all the attitude types that it supports:
    - MRPs
    - Euler Parameters (quaternion) [q0, q1, q2, q3]
    - Euler Parameters (quaternion) [q1, q2, q3, qs]
    and with angular rates and accelerations that can be expressed either in the inertial frame N or in the 
    reference frame R.
    This unit test writes 5 time-tagged attitudes at times t = [1.0, 2.0, 3.0, 4.0, 5.0]s. Real values of 
    attitude parameters, angular rates and angular accelerations in inertial frames are stored in 
    ``attReal_RN``, ``omegaReal_RN_N`` and ``omegaDotReal_RN_N`` respectively.

    **Test Parameters**

    Args:
        attType (int): 0 - MRPs; 1 - EP [q0, q1, q2, q3]; 2 - [q1, q2, q3, qs]
        MRPswitching : False: every waypoint is within 180 deg from the inertial frame; True: some waipoints exceed 180 deg rotation from the inertial frame and ``attRefOutMsg.sigma_RN`` presents a discontinuity in correspondence of MRP switching;
        useReferenceFrame (bool): False: ang. rates and accelerations expressed in inertial frame; True: ang. rates and accelerations expressed in reference frame;
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    This unit test checks the correctness of the output attitude reference message 

    - ``attRefMsg``

    compared to the real values stored in the data file  ``attReal_RN``, ``omegaReal_RN_N`` and ``omegaDotReal_RN_N``.
    The simulation is run with a sampling frequency of 0.25 s, which is higher than the frequency with which the attitude
    waypoints are saved in the data file (1.0 s), starting at t = 0. 

    For t < 1.0 s we check that the attitude in ``attRefMsg`` coincides with ``attReal_RN`` at time t = 1.0 s, 
    while rates and accelerations in ``attRefMsg`` are zero.

    For t > 5.0 s we check that the attitude in ``attRefMsg`` coincides with ``attReal_RN`` at time t = 5.0 s, 
    while rates and accelerations in ``attRefMsg`` are zero.

    For 1.0 s <= t <= 5.0 s we check that the attitude, rates and accelerations in ``attRefMsg`` coincide with 
    the linear interpolation of ``attReal_RN``, ``omegaReal_RN_N`` and ``omegaDotReal_RN_N``.
    """

    # each test method requires a single assert method to be called
    [testResults, testMessage] = waypointReferenceTestFunction(attType, MRPswitching, useReferenceFrame, accuracy)
    global dataFileName
    if os.path.exists(dataFileName):
        os.remove(dataFileName)
    assert testResults < 1, testMessage

def waypointReferenceTestFunction(attType, MRPswitching, useReferenceFrame, accuracy):

    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    dtSeconds = 0.25
    simTimeSeconds = 6
    testProcessRate = macros.sec2nano(dtSeconds)
    simulationTime = macros.sec2nano(simTimeSeconds)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # create the simulation data file
    # dataFileName
    global dataFileName
    dataFileName = "data" + str(attType) + str(int(MRPswitching))
    if useReferenceFrame == True:
        dataFileName += "R.txt"
    else:
        dataFileName += "N.txt"

    dataFileName = os.path.join(path, dataFileName)
    delimiter = ","
    fDataFile = open(dataFileName, "w+")
    
    # create the datafile and store the real attitude, rate and acceleration values
    t = []
    attReal_RN = []
    omegaReal_RN_N = []
    omegaDotReal_RN_N = []
    for i in range(0, 5):
        t.append(i + 1)
        if MRPswitching:
            s = 0.8
        else:
            s = 0.1
        attRealMRP = np.array([s + 0.05*i, 0.2 + 0.05*i, 0.3 + 0.05*i])
        if np.linalg.norm(attRealMRP) <= 1:
            attReal_RN.append(attRealMRP)
        else:
            attReal_RN.append(-attRealMRP / (np.linalg.norm(attRealMRP))**2 )
        omegaReal_RN_N.append(np.array([0.4 + 0.05*i, 0.5 + 0.05*i, 0.6 + 0.05*i]))
        omegaDotReal_RN_N.append(np.array([0.7 + 0.05*i, 0.8 + 0.05*i, 0.9 + 0.05*i]))

        lineString = str(t[-1]) + delimiter

        if attType == 0:
            lineString += str(attReal_RN[-1].tolist())[1:-1] + delimiter
        elif attType == 1:
            q = rbk.MRP2EP(attReal_RN[-1])
            lineString += str(q.tolist())[1:-1] + delimiter
        elif attType == 2:
            q = rbk.MRP2EP(attReal_RN[-1])
            qs = [q[1], q[2], q[3], q[0]]
            lineString += str(qs)[1:-1] + delimiter
        else:
            print("Invalid attitude type")
            return
        
        if not useReferenceFrame:
            lineString += str(omegaReal_RN_N[-1].tolist())[1:-1] + delimiter + str(omegaDotReal_RN_N[-1].tolist())[1:-1] + '\n'
        else:
            RN = rbk.MRP2C(attReal_RN[-1])
            omegaReal_RN_R = np.matmul(RN, omegaReal_RN_N[-1])
            omegaDotReal_RN_R = np.matmul(RN, omegaDotReal_RN_N[-1])
            lineString += str(omegaReal_RN_R.tolist())[1:-1] + delimiter + str(omegaDotReal_RN_R.tolist())[1:-1] + '\n'
        
        # write line on file
        fDataFile.write(lineString)

    # close file
    fDataFile.close()

    # Construct algorithm and associated C++ container
    testModule = waypointReference.WaypointReference()
    testModule.ModelTag = "testModule"

    # load the data path from the same folder where this python script is
    testModule.dataFileName = dataFileName
    testModule.delimiter = delimiter
    testModule.attitudeType = attType
    testModule.useReferenceFrame = useReferenceFrame
    testModule.headerLines = 0

    # Add module to the task
    unitTestSim.AddModelToTask(unitTaskName, testModule)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = testModule.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(simulationTime) 

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the sampling times from the simulation run.
    timeData = dataLog.times() * macros.NANO2SEC
    
    # Check if logged data matches the real attitude, rates and accelerations
    j = 0
    
    sigma_RN = [[], [], []]
    # checking attitude msg for t < t_min
    for i in range(len(timeData)-1):

        for n in range(3):
            sigma_RN[n].append(dataLog.sigma_RN[i][n])

        if timeData[i] < t[0]:
            if not unitTestSupport.isVectorEqual(dataLog.sigma_RN[i], attReal_RN[0], accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed attitude check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.omega_RN_N[i], np.array([0.0, 0.0, 0.0]), accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular rate check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.domega_RN_N[i], np.array([0.0, 0.0, 0.0]), accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular acceleration check at time t = {}".format(timeData[i]))
        
        # checking attitude msg for t_min <= t <= t_max
        elif timeData[i] >= t[0] and timeData[i] <= t[-1]:
            while (timeData[i] >= t[j] and timeData[i] <= t[j+1]) == False:
                j += 1
            sigma_RN_int = np.array([0.0, 0.0, 0.0])
            omega_RN_N_int = np.array([0.0, 0.0, 0.0])
            omegaDot_RN_N_int = np.array([0.0, 0.0, 0.0])

            # interpolating between attitudes for times t = timeData[i]
            if np.linalg.norm( attReal_RN[j+1] - attReal_RN[j] ) <= 1:
                sigma_RN_int = attReal_RN[j] + (attReal_RN[j+1] - attReal_RN[j]) / (t[j+1] - t[j]) * (timeData[i] - t[j])
            else:
                attReal_RN_SS = -attReal_RN[j+1] / (np.linalg.norm(attReal_RN[j+1]))**2
                sigma_RN_int = attReal_RN[j] + (attReal_RN_SS - attReal_RN[j]) / (t[j+1] - t[j]) * (timeData[i] - t[j])
            omega_RN_N_int = omegaReal_RN_N[j] + (omegaReal_RN_N[j+1] - omegaReal_RN_N[j]) / (t[j+1] - t[j]) * (timeData[i] - t[j])
            omegaDot_RN_N_int = omegaDotReal_RN_N[j] + (omegaDotReal_RN_N[j+1] - omegaDotReal_RN_N[j]) / (t[j+1] - t[j]) * (timeData[i] - t[j])
            if not unitTestSupport.isVectorEqual(dataLog.sigma_RN[i], sigma_RN_int, accuracy):
                print(timeData[i], dataLog.sigma_RN[i], sigma_RN_int)
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed attitude check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.omega_RN_N[i], omega_RN_N_int, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular rate check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.domega_RN_N[i], omegaDot_RN_N_int, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular acceleration check at time t = {}".format(timeData[i]))
        
        # checking attitude msg for t < t_max
        else:
            if not unitTestSupport.isVectorEqual(dataLog.sigma_RN[i], attReal_RN[-1], accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed attitude check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.omega_RN_N[i], [0.0, 0.0, 0.0], accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular rate check at time t = {}".format(timeData[i]))
            if not unitTestSupport.isVectorEqual(dataLog.domega_RN_N[i], [0.0, 0.0, 0.0], accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + testModule.ModelTag + " Module failed angular acceleration check at time t = {}".format(timeData[i]))

    # print out success or failure message
    if testFailCount == 0:
        print("PASSED: " + testModule.ModelTag)
    else:
        print("FAILED: " + testModule.ModelTag)
        print(testMessages)

    return [testFailCount, ''.join(testMessages)]

#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    waypointReferenceTestFunction(
        2,        # attType (0 -> MRP, 1 -> quaternion [q0, q1, q2, q3], 2 -> quaternion [q1, q2, q3, qs])
        False,    # MRPswitching
        False,    # useReferenceFrame
        1e-12)
    if os.path.exists(dataFileName):
        os.remove(dataFileName)
	   