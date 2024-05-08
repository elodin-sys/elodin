#
#   Unit Test Script
#   Module Name:        vehicleConfigData
#   Creation Date:      October 5, 2018
#

from Basilisk.fswAlgorithms import vehicleConfigData
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


def test_vehicleConfigData():
    """Module Unit Test"""
    [testResults, testMessage] = vehicleConfigDataTestFunction()

    assert testResults < 1, testMessage

def vehicleConfigDataTestFunction():
    """ Test the vehicleConfigData module """

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate)) # Add a new task to the process

    # Construct the cssComm module
    module = vehicleConfigData.vehicleConfigData()
    # Populate the config
    I = [1000., 0., 0.,
         0., 800., 0.,
         0., 0., 800.]
    module.ISCPntB_B = I
    initialCoM = [1, 1, 1]
    module.CoM_B = initialCoM
    mass = 300.
    module.massSC = mass

    module.ModelTag = "vehicleConfigData"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Log the output message
    dataLog = module.vecConfigOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    unitTestSim.ConfigureStopTime(testProcessRate)
    unitTestSim.ExecuteSimulation()

    # Get the output from this simulation
    Ilog = dataLog.ISCPntB_B
    CoMLog = dataLog.CoM_B
    MassLog = dataLog.massSC

    accuracy = 1e-6

    # At each timestep, make sure the vehicleConfig values haven't changed from the initial values
    testFailCount, testMessages = unitTestSupport.compareArrayND([initialCoM for _ in range(len(CoMLog))], CoMLog, accuracy,
                                                                 "VehicleConfigData CoM",
                                                                 3, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArrayND([I for _ in range(len(Ilog))], Ilog, accuracy,
                                                                 "VehicleConfigData I",
                                                                 3, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareDoubleArray([mass for _ in range(len(MassLog))], MassLog, accuracy,
                                                                 "VehicleConfigData Mass",
                                                                 testFailCount, testMessages)

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, ''.join(testMessages)]

if __name__ == '__main__':
    test_vehicleConfigData()
