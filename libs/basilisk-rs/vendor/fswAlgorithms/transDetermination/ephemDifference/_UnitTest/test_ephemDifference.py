#
#   Unit Test Script
#   Module Name:        ephemDifference
#   Creation Date:      October 16, 2018
#


import inspect
import os
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros
from Basilisk.fswAlgorithms import ephemDifference
from Basilisk.utilities import astroFunctions
from Basilisk.architecture import messaging

@pytest.mark.parametrize("ephBdyCount", [3, 0])

def test_ephemDifference(ephBdyCount):
    """ Test ephemDifference. """
    [testResults, testMessage] = ephemDifferenceTestFunction(ephBdyCount)
    assert testResults < 1, testMessage

def ephemDifferenceTestFunction(ephBdyCount):
    """ Test the ephemDifference module. Setup a simulation, """

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

    ephemDiff = ephemDifference.ephemDifference()

    # This calls the algContain to setup the selfInit, update, and reset
    ephemDiff.ModelTag = "ephemDifference"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, ephemDiff)

    # Create the input message.
    inputEphemBase = messaging.EphemerisMsgPayload() # The clock correlation message ?
    # Get the Earth's position and velocity
    position, velocity = astroFunctions.Earth_RV(astroFunctions.JulianDate([2018, 10, 16]))
    inputEphemBase.r_BdyZero_N = position
    inputEphemBase.v_BdyZero_N = velocity
    inputEphemBase.timeTag = 1234.0
    ephBaseInMsg = messaging.EphemerisMsg().write(inputEphemBase)
    ephemDiff.ephBaseInMsg.subscribeTo(ephBaseInMsg)
    functions = [astroFunctions.Mars_RV, astroFunctions.Jupiter_RV, astroFunctions.Saturn_RV]

    changeBodyList = list()
    ephInMsgList = list()
    if ephBdyCount == 3:
        for i in range(ephBdyCount):
            # Create the change body message
            changeBodyMsg = ephemDifference.EphemChangeConfig()

            changeBodyList.append(changeBodyMsg)

            # Create the input message to the change body config
            inputMsg = messaging.EphemerisMsgPayload()
            position, velocity = functions[i](astroFunctions.JulianDate([2018, 10, 16]))
            inputMsg.r_BdyZero_N = position
            inputMsg.v_BdyZero_N = velocity
            inputMsg.timeTag = 321.0

            # Set this message
            ephInMsgList.append(messaging.EphemerisMsg().write(inputMsg))
            changeBodyMsg.ephInMsg.subscribeTo(ephInMsgList[-1])

    ephemDiff.changeBodies = changeBodyList

    # the logging setup must occur on the actual ephemDiff.changeBodies[i].ephOutMsg as we are providing
    # pointers to the message payload.  Logging changeBodyList.ephOutMsg won't work as this message has a
    # different location.
    dataLogList = list()
    for i in range(ephBdyCount):
        dataLogList.append(ephemDiff.changeBodies[i].ephOutMsg.recorder())
        unitTestSim.AddModelToTask(unitTaskName, dataLogList[i])

    # Initialize the simulation
    unitTestSim.InitializeSimulation()

    # The result isn't going to change with more time. The module will continue to produce the same result
    unitTestSim.ConfigureStopTime(0)  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    if ephBdyCount == 3:
        trueRVector = [[69313607.6209608,  -75620898.04028425,   -5443274.17030424],
                       [-5.33462105e+08,  -7.56888610e+08,   1.17556184e+07],
                       [9.94135029e+07,  -1.54721593e+09,   1.65081472e+07]]

        trueVVector = [[15.04232523,  -1.13359121,   0.47668898],
                       [23.2531093,  -33.17628299,  -0.22550391],
                       [21.02793499, -25.86425597,  -0.38273815]]


        posAcc = 1e1
        velAcc = 1e-4
        unitTestSupport.writeTeXSnippet("toleranceValuePos", str(posAcc), path)
        unitTestSupport.writeTeXSnippet("toleranceValueVel", str(velAcc), path)

        for i in range(ephBdyCount):

            outputData_R = dataLogList[i].r_BdyZero_N
            outputData_V = dataLogList[i].v_BdyZero_N
            timeTag = dataLogList[i].timeTag
            # print(timeTag)
            # print(outputData_R)

            # At each timestep, make sure the vehicleConfig values haven't changed from the initial values
            testFailCount, testMessages = unitTestSupport.compareArrayND([trueRVector[i]], outputData_R,
                                                                         posAcc,
                                                                         "ephemDifference position output body " + str(i),
                                                                         2, testFailCount, testMessages)
            testFailCount, testMessages = unitTestSupport.compareArrayND([trueVVector[i]], outputData_V,
                                                                         velAcc,
                                                                         "ephemDifference velocity output body " + str(i),
                                                                         2, testFailCount, testMessages)
            if timeTag[0] != 321.0:
                testFailCount += 1
                testMessages.append("ephemDifference timeTag output body " + str(i))

    if ephemDiff.ephBdyCount is not ephBdyCount:
        testFailCount += 1
        testMessages.append("input/output message count is wrong.")

    snippentName = "passFail" + str(ephBdyCount)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + ephemDiff.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + ephemDiff.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)

    return [testFailCount, ''.join(testMessages)]


if __name__ == '__main__':
    test_ephemDifference(3)
