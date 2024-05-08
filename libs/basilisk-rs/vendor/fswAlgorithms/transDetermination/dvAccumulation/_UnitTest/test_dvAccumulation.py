#
#   Unit Test Script
#   Module Name:        dvAccumulation
#   Creation Date:      October 5, 2018
#

import inspect
import os

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import dvAccumulation
from Basilisk.utilities import SimulationBaseClass, unitTestSupport
from Basilisk.utilities import macros
from numpy import random

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))


def generateAccData():
    """ Returns a list of random AccPktDataFswMsg."""
    accPktList = list()
    for _ in range(120):
        accPacketData = messaging.AccPktDataMsgPayload()
        accPacketData.measTime = abs(int(random.normal(5e7, 1e7)))
        accPacketData.accel_B = random.normal(0.1, 0.2, 3)  # Acceleration in platform frame [m/s2]
        accPktList.append(accPacketData)

    return accPktList

def test_dv_accumulation():
    """ Test dvAccumulation. """
    [testResults, testMessage] = dvAccumulationTestFunction()
    assert testResults < 1, testMessage

def dvAccumulationTestFunction():
    """ Test the dvAccumulation module. Setup a simulation, """

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Test quicksort routine
    # Generate (1) random packet measurement times and (2) completely inverted measurement times
    randMeasTimes = []
    invMeasTimes = []
    randData = messaging.AccDataMsgPayload()
    invData = messaging.AccDataMsgPayload()
    for i in range(0, messaging.MAX_ACC_BUF_PKT):
        randMeasTimes.append(random.randint(0, 1000000))
        randData.accPkts[i].measTime = randMeasTimes[i]

        invMeasTimes.append(messaging.MAX_ACC_BUF_PKT - i)
        invData.accPkts[i].measTime = invMeasTimes[i]

    # Run module quicksort function
    dvAccumulation.dvAccumulation_QuickSort(randData.accPkts[0], 0, messaging.MAX_ACC_BUF_PKT - 1)
    dvAccumulation.dvAccumulation_QuickSort(invData.accPkts[0], 0, messaging.MAX_ACC_BUF_PKT - 1)

    # Check that sorted packets properly
    randMeasTimes.sort()
    invMeasTimes.sort()
    for i in range(0, messaging.MAX_ACC_BUF_PKT):
        if randData.accPkts[i].measTime != randMeasTimes[i]:
            testFailCount += 1
        if invData.accPkts[i].measTime != invMeasTimes[i]:
            testFailCount += 1

    # Test Module
    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # This is needed if multiple unit test scripts are run
    # This create a fresh and consistent simulation environment for each test run

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))  # Add a new task to the process

    # Construct the dvAccumulation module
    # Set the names for the input messages
    module = dvAccumulation.dvAccumulation()

    # This calls the algContain to setup the selfInit, update, and reset
    module.ModelTag = "dvAccumulation"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, module)

    dataLog = module.dvAcumOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Create the input message.
    inputAccData = messaging.AccDataMsgPayload()

    # Set this as the packet data in the acceleration data
    random.seed(12345)
    inputAccData.accPkts = generateAccData()
    inMsg = messaging.AccDataMsg()
    module.accPktInMsg.subscribeTo(inMsg)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()
    inMsg.write(inputAccData)

    #   Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # Create the input message again to simulate multiple acceleration inputs.
    inputAccData = messaging.AccDataMsgPayload()

    # Set this as the packet data in the acceleration data. Test the module with different inputs.
    inputAccData.accPkts = generateAccData()

    # Write this message
    inMsg.write(inputAccData)

    #   Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    outputNavMsgData = dataLog.vehAccumDV
    timeMsgData = dataLog.timeTag

    # print(outputNavMsgData)
    # print(timeMsgData)

    trueDVVector = [[4.82820079e-03,   7.81971465e-03,   2.29605663e-03],
                 [ 4.82820079e-03,   7.81971465e-03,   2.29605663e-03],
                 [ 4.82820079e-03,   7.81971465e-03,   2.29605663e-03],
                 [ 6.44596343e-03,   9.00203561e-03,   2.60580728e-03],
                 [ 6.44596343e-03,   9.00203561e-03,   2.60580728e-03]]
    trueTime = np.array([7.2123026e+07, 7.2123026e+07, 7.2123026e+07, 7.6667436e+07, 7.6667436e+07]) * macros.NANO2SEC

    accuracy = 1e-6
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)

    # At each timestep, make sure the vehicleConfig values haven't changed from the initial values
    testFailCount, testMessages = unitTestSupport.compareArrayND(trueDVVector, outputNavMsgData,
                                                                 accuracy,
                                                                 "dvAccumulation output",
                                                                 2, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArrayND([trueTime], [timeMsgData],
                                                                 accuracy, "timeTag", 5,
                                                                 testFailCount, testMessages)

    snippentName = "passFail"
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)

    return [testFailCount, ''.join(testMessages)]

if __name__ == '__main__':
    test_dv_accumulation()
