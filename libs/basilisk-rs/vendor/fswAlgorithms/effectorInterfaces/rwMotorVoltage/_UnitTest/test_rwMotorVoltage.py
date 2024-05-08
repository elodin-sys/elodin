# ISC License
#
# Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


#
#   Unit Test Script
#   Module Name:        rwMotorVoltage
#   Author:             Hanspeter Schaub
#   Creation Date:      January 16, 2017
#

import inspect
import os

import numpy as np
import pytest

# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport                  # general support file with common unit test functions
from Basilisk.fswAlgorithms import rwMotorVoltage
from Basilisk.utilities import fswSetupRW
from Basilisk.utilities import macros
from Basilisk.architecture import messaging


filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

def addTimeColumn(time, data):
    return np.transpose(np.vstack([[time], np.transpose(data)]))

# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("useLargeVoltage, useAvailability, useTorqueLoop, testName", [
       (False, False, False, "One")
     , (True, False, False, "Two")
     , (False, True, False, "Three")
     , (False, False, True, "Four")
])

# update "module" in this function name to reflect the module name
def test_module(show_plots, useLargeVoltage, useAvailability, useTorqueLoop, testName):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = run(show_plots, useLargeVoltage, useAvailability, useTorqueLoop, testName)
    assert testResults < 1, testMessage


def run(show_plots, useLargeVoltage, useAvailability, useTorqueLoop, testName):
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
    module = rwMotorVoltage.rwMotorVoltage()
    module.ModelTag = "rwMotorVoltage"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Initialize the test module configuration data
    # set module parameters
    module.VMin = 1.0     # Volts
    module.VMax = 11.0    # Volts

    if useTorqueLoop:
        module.K = 1.5
        rwSpeedMessage = messaging.RWSpeedMsgPayload()
        rwSpeedMessage.wheelSpeeds = [1.0, 2.0, 1.5, -3.0]      # rad/sec Omega's
        rwSpeedInMsg = messaging.RWSpeedMsg().write(rwSpeedMessage)
        module.rwSpeedInMsg.subscribeTo(rwSpeedInMsg)
        unitTestSupport.writeTeXSnippet("Omega1", r"$\bm\Omega = " \
                                        + str(rwSpeedMessage.wheelSpeeds[0:4]) + "$"
                                        , path)

    #
    #   create BSK messages
    #
    # Create RW configuration parameter input message
    GsMatrix_B = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0]         # the create routine below normalizes these vectors
    ]
    fswSetupRW.clearSetup()
    for i in range(4):
        fswSetupRW.create(GsMatrix_B[i],    #           spin axis
                          0.1,              # kg*m^2    J2
                          0.2)              # Nm        uMax
    rwConfigInMsg = fswSetupRW.writeConfigMessage()
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    numRW = fswSetupRW.getNumOfDevices()

    # Create RW motor torque input message
    usMessageData = messaging.ArrayMotorTorqueMsgPayload()
    if useLargeVoltage:
        usMessageData.motorTorque = [0.5, 0.0, -0.15, -0.5]           # [Nm] RW motor torque cmds
    else:
        usMessageData.motorTorque = [0.05, 0.0, -0.15, -0.2]  # [Nm] RW motor torque cmds
    rwMotorTorqueInMsg = messaging.ArrayMotorTorqueMsg().write(usMessageData)
    module.torqueInMsg.subscribeTo(rwMotorTorqueInMsg)

    # create RW availability message
    if useAvailability:
        rwAvailabilityMessage = messaging.RWAvailabilityMsgPayload()
        rwAvailArray = np.zeros(messaging.MAX_EFF_CNT, dtype=int)
        rwAvailArray.fill(messaging.AVAILABLE)
        rwAvailArray[2] = messaging.UNAVAILABLE        # make 3rd RW unavailable
        rwAvailabilityMessage.wheelAvailability = rwAvailArray
        rwAvailInMsg = messaging.RWAvailabilityMsg().write(rwAvailabilityMessage)
        module.rwAvailInMsg.subscribeTo(rwAvailInMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.voltageOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    if useTorqueLoop:
        rwSpeedMessage.wheelSpeeds = [1.1, 2.1, 1.1, -4.1]  # rad/sec Omega's
        rwSpeedInMsg.write(rwSpeedMessage)
        unitTestSupport.writeTeXSnippet("Omega2", r"$\bm\Omega = " \
                                        + str(rwSpeedMessage.wheelSpeeds[0:4]) + "$"
                                        , path)
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.5))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # reset the module to test this functionality
    module.Reset(1)     # this module reset function needs a time input (in NanoSeconds)

    # run the module again for an additional 1.0 seconds
    unitTestSim.ConfigureStopTime(macros.sec2nano(3.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()


    # This pulls the actual data log from the simulation run.
    moduleOutput = dataLog.voltage[:, :numRW]
    print(moduleOutput)


    # set the filtered output truth states
    trueVector=[];
    if not useLargeVoltage and not useAvailability and not useTorqueLoop:
        trueVector = [
                   [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                   ]
    if useLargeVoltage and not useAvailability and not useTorqueLoop:
        trueVector = [
                   [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                 , [11., 0., -8.5, -11.]
                   ]
    if not useLargeVoltage and useAvailability and not useTorqueLoop:
        trueVector = [
                   [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                 , [3.5, 0., 0., -11.]
                   ]
    if not useLargeVoltage and not useAvailability and useTorqueLoop:
        trueVector = [
                   [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [5.75, -2.5, -11., -9.5]
                 , [3.5, 0., -8.5, -11.]
                 , [3.5, 0., -8.5, -11.]
                 , [7.25, 0., -11., -11.]
                   ]

    # compare the module results to the truth values
    accuracy = 1e-10

    testFailCount, testMessages = unitTestSupport.compareArray(trueVector, moduleOutput,
                                                               accuracy, "Output Vector",
                                                               testFailCount, testMessages)



    # If the argument provided at commandline "--show_plots" evaluates as true,
    # plot all figures
    # plot a sample variable.
    # plt.close("all")    # close all prior figures so we start with a clean slate
    # plt.figure(1)
    # plt.plot(variableState[:, 0]*macros.NANO2SEC, variableState[:, 1],
    #          label='Case useLargeVoltage = ' + str(useLargeVoltage))
    # plt.legend(loc='upper left')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Variable Description [unit]')
    # if show_plots:
    #     plt.show()
    #     plt.close('all')

    #   print out success message if no error were found
    snippentName = "passFail" + testName
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)

    # write TeX Tables for documentation
    moduleOutput = addTimeColumn(dataLog.times(), dataLog.voltage)[:, :numRW+1]
    resultTable = moduleOutput
    resultTable[:, 0] = macros.NANO2SEC * resultTable[:, 0]
    diff = np.delete(moduleOutput, 0, 1) - trueVector
    resultTable = np.insert(resultTable, list(range(2, 2 + len(diff.transpose()))), diff, axis=1)

    tableName = "test" + str(useLargeVoltage) + str(useAvailability) + str(useTorqueLoop)
    tableHeaders = ["time [s]", "$V_{s,1}$", "Error", "$V_{s,2}$", "Error", "$V_{s,3}$", "Error", "$V_{s,4}$", "Error"]
    caption = 'RW voltage output for case {\\tt useLargeVoltage = ' + str(useLargeVoltage) \
              + ', useAvailability = ' + str(useAvailability) \
              + ', useTorqueLoop = ' + str(useTorqueLoop) + '}.'
    unitTestSupport.writeTableLaTeX(
        tableName,
        tableHeaders,
        caption,
        resultTable,
        path)
    unitTestSupport.writeTeXSnippet("us"+ str(useLargeVoltage) + str(useAvailability) + str(useTorqueLoop)
                                    , "$\\bm u_s = " + str(usMessageData.motorTorque[0:numRW]) + "$"
                                    , path)

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_module(              # update "module" in function name
                  False
                 ,False       # useLargeVoltage
                 ,False       # useAvailability
                 ,True        # useTorqueLoop
                 ,"Four"      # testName
               )
