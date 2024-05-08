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
#   Module Name:        sunlineEphem()
#   Author:             John Martin
#   Creation Date:      November 30, 2018
#

import matplotlib.pyplot as plt
import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import sunlineEphem  # import the module that is to be tested
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.


class DataStore:
    """Container for developer defined variables to be used in test data post-processing and plotting.

        Attributes:
            variableState (list): an example variable to hold test result data.
    """

    def __init__(self):
        self.variableState = None  # replace/add with appropriate variables for test result data storing

    def plotData(self):
        """All test plotting to be performed here.

        """
        plt.figure(1) # plot a sample variable.
        plt.plot(self.variableState[:, 0]*macros.NANO2SEC, self.variableState[:, 1], label='Sample Variable')
        plt.legend(loc='upper left')
        plt.xlabel('Time [s]')
        plt.ylabel('Variable Description [unit]')
        plt.show()


@pytest.fixture(scope="module")
def plotFixture(show_plots):
    dataStore = DataStore()
    yield dataStore
    if show_plots:
        dataStore.plotData()


# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_
def test_module(show_plots):     # update "module" in this function name to reflect the module name
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = sunlineEphemTestFunction(show_plots)
    assert testResults < 1, testMessage


def sunlineEphemTestFunction(show_plots):
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C++ container
    sunlineEphemObj = sunlineEphem.sunlineEphem()
    sunlineEphemObj.ModelTag = "sunlineEphem"           # update python name of test module

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, sunlineEphemObj)

    # Create input message and size it because the regular creator of that message
    # is not part of the test.

    vehAttData = messaging.NavAttMsgPayload()
    vehPosData = messaging.NavTransMsgPayload()
    sunData = messaging.EphemerisMsgPayload()


    # Artificially put sun at the origin.
    sunData.r_BdyZero_N = [0.0, 0.0, 0.0]
    vehAttInMsg = messaging.NavAttMsg().write(vehAttData)


    # Place spacecraft unit length away on each coordinate axis
    vehAttData.sigma_BN = [0.0, 0.0, 0.0]
    TestVectors = [[-1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]]

    estVector = np.zeros((6, 3))

    vehPosInMsg = messaging.NavTransMsg()
    sunDataInMsg = messaging.EphemerisMsg().write(sunData)
    sunlineEphemObj.sunPositionInMsg.subscribeTo(sunDataInMsg)
    sunlineEphemObj.scPositionInMsg.subscribeTo(vehPosInMsg)
    sunlineEphemObj.scAttitudeInMsg.subscribeTo(vehAttInMsg)

    dataLog = sunlineEphemObj.navStateOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    for i in range(len(TestVectors)):
        testVec = TestVectors[i]
        vehPosData.r_BN_N = testVec
        vehPosInMsg.write(vehPosData)

        # Need to call the self-init and cross-init methods
        unitTestSim.InitializeSimulation()
        unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation
        unitTestSim.ExecuteSimulation()
        estVector[i] = dataLog.vehSunPntBdy[-1]

        # reset the module to test this functionality
        sunlineEphemObj.Reset(1)


    # set the filtered output truth states
    trueVector = [
               [1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0],
               [-1.0, 0.0, 0.0],
               [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
               ]

    # compare the module results to the truth values
    accuracy = 1e-12
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(estVector[i], trueVector[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + sunlineEphemObj.ModelTag + " Module failed sunlineEphem " +
                                " unit test at t=" + str(dataLog.times()[i]*macros.NANO2SEC) + "sec\n")




    #   print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + sunlineEphemObj.ModelTag)
    else:
        print(testMessages)

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_module(            # update "subModule" in function name
               False        # show_plots
    )
