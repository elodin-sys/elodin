#
#   Unit Test Script
#   Module Name:        rwConfigData
#   Creation Date:      October 5, 2018
#

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import rwConfigData
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


def test_rwConfigData():
    """Module Unit Test"""
    [testResults, testMessage] = rwConfigDataTestFunction()
    assert testResults < 1, testMessage

def rwConfigDataTestFunction():
    """ Test the rwConfigData module """

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # This is needed if multiple unit test scripts are run
    # This create a fresh and consistent simulation environment for each test run

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate)) # Add a new task to the process

    # Construct the cssComm module
    module = rwConfigData.rwConfigData()

    # Create the messages
    rwConstellationFswMsg = messaging.RWConstellationMsgPayload()
    numRW = 3
    rwConstellationFswMsg.numRW = 3
    gsHat_initial = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    js_initial = np.array([0.08, 0.09, 0.07])
    uMax_initial = np.array([0.2, 0.1, 0.3])

    # Iterate over all of the reaction wheels, create a rwConfigElementFswMsg, and add them to the rwConstellationFswMsg
    rwConfigElementList = list()
    for rw in range(numRW):
        rwConfigElementMsg = messaging.RWConfigElementMsgPayload()
        rwConfigElementMsg.gsHat_B = gsHat_initial[rw]  # Spin axis unit vector of the wheel in structure # [1, 0, 0]
        rwConfigElementMsg.Js = js_initial[rw]  # Spin axis inertia of wheel [kgm2]
        rwConfigElementMsg.uMax = uMax_initial[rw]  # maximum RW motor torque [Nm]

        # Add this to the list
        rwConfigElementList.append(rwConfigElementMsg)

    # Set the array of the reaction wheels in RWConstellationFswMsg to the list created above
    rwConstellationFswMsg.reactionWheels = rwConfigElementList

    # Set these messages
    rwConstInMsg = messaging.RWConstellationMsg().write(rwConstellationFswMsg)
    module.rwConstellationInMsg.subscribeTo(rwConstInMsg)

    module.ModelTag = "rwConfigData"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Log the output message
    dataLog = module.rwParamsOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    unitTestSim.ConfigureStopTime(testProcessRate)
    unitTestSim.ExecuteSimulation()

    # Get the output from this simulation
    JsListLog = dataLog.JsList[:, :numRW]
    uMaxLog = dataLog.uMax[:, :numRW]
    GsMatrix_B_Log = dataLog.GsMatrix_B[:, :(3*numRW)]

    accuracy = 1e-6
    # At each timestep, make sure the vehicleConfig values haven't changed from the initial values
    testFailCount, testMessages = unitTestSupport.compareArrayND([js_initial]*2, JsListLog, accuracy,
                                                                 "rwConfigData JsList",
                                                                 3, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArrayND([uMax_initial]*2, uMaxLog, accuracy,
                                                                 "rwConfigData uMax",
                                                                 3, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArrayND([gsHat_initial.flatten()]*2, GsMatrix_B_Log, accuracy,
                                                                 "rwConfigData GsMatrix_B",
                                                                 3*numRW, testFailCount, testMessages)

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)

    return [testFailCount, ''.join(testMessages)]

if __name__ == '__main__':
    test_rwConfigData()
