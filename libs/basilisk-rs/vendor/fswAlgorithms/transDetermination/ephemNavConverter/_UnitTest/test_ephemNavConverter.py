#
#   Unit Test Script
#   Module Name:        ephemNavConverter
#   Creation Date:      October 16, 2018
#

import inspect
import os

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import ephemNavConverter
from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros
from Basilisk.utilities import astroFunctions

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

def test_ephemNavConverter():
    """ Test ephemNavConverter. """
    [testResults, testMessage] = ephemNavConverterTestFunction()
    assert testResults < 1, testMessage

def ephemNavConverterTestFunction():
    """ Test the ephemNavConverter module. Setup a simulation """

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))  # Add a new task to the process

    # Construct the ephemNavConverter module
    # Set the names for the input messages
    ephemNav = ephemNavConverter.ephemNavConverter()

    # This calls the algContain to setup the selfInit, update, and reset
    ephemNav.ModelTag = "ephemNavConverter"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, ephemNav)

    # Create the input message.
    inputEphem = messaging.EphemerisMsgPayload()

    # Get the Earth's position and velocity
    position, velocity = astroFunctions.Earth_RV(astroFunctions.JulianDate([2018, 10, 16]))
    inputEphem.r_BdyZero_N = position
    inputEphem.v_BdyZero_N = velocity
    inputEphem.timeTag = 1.0  # sec
    inMsg = messaging.EphemerisMsg().write(inputEphem)
    ephemNav.ephInMsg.subscribeTo(inMsg)

    dataLog = ephemNav.stateOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    # The result isn't going to change with more time. The module will continue to produce the same result
    unitTestSim.ConfigureStopTime(testProcessRate)  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    posAcc = 1e1
    velAcc = 1e-4

    outputR = dataLog.r_BN_N
    outputV = dataLog.v_BN_N
    outputTime = dataLog.timeTag

    trueR = [position, position]
    trueV = [velocity, velocity]
    trueTime = [inputEphem.timeTag, inputEphem.timeTag]

    # At each timestep, make sure the vehicleConfig values haven't changed from the initial values
    testFailCount, testMessages = unitTestSupport.compareArrayND(trueR, outputR,
                                                                 posAcc,
                                                                 "ephemNavConverter output Position",
                                                                 2, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArrayND(trueV, outputV,
                                                                 velAcc,
                                                                 "ephemNavConverter output Velocity",
                                                                 2, testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareDoubleArray(trueTime, outputTime,
                                                                 velAcc,
                                                                 "ephemNavConverter output Time",
                                                                 testFailCount, testMessages)

    #   print out success message if no error were found
    snippentName = "passFail"
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + ephemNav.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + ephemNav.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)


    return [testFailCount, ''.join(testMessages)]


if __name__ == '__main__':
    test_ephemNavConverter()
