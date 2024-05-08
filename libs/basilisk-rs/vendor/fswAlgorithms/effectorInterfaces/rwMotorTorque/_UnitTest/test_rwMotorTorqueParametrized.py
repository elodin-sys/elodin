#
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
#   Module Name:        rwMotorTorque
#   Author:             Hanspeter Schaub
#   Creation Date:      July 4, 2016
#

import inspect
import os

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))




# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.fswAlgorithms import rwMotorTorque
from Basilisk.utilities import macros
from Basilisk.architecture import messaging
from Support import results_rwMotorTorque

# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("numControlAxes", [0, 1, 2, 3])
@pytest.mark.parametrize("numWheels", [2, 4, messaging.MAX_EFF_CNT])
@pytest.mark.parametrize("RWAvailMsg",["NO", "ON", "OFF", "MIXED"])


# update "module" in this function name to reflect the module name
def test_rwMotorTorque(show_plots, numControlAxes, numWheels, RWAvailMsg):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = rwMotorTorqueTest(show_plots, numControlAxes, numWheels, RWAvailMsg)
    assert testResults < 1, testMessage


def rwMotorTorqueTest(show_plots, numControlAxes, numWheels, RWAvailMsg):
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
    module = rwMotorTorque.rwMotorTorque()
    module.ModelTag = "rwMotorTorque"


    # Initialize module variables
    if numControlAxes == 3:
        controlAxes_B = [
            1, 0, 0
            , 0, 1, 0
            , 0, 0, 1
        ]
    elif numControlAxes == 2:
        controlAxes_B = [
             1,0,0
            ,0,1,0
        ]
    elif numControlAxes == 1:
        controlAxes_B = [
            1, 0, 0
        ]
    else:
        controlAxes_B = []

    module.controlAxes_B = controlAxes_B


    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)


    # attControl message
    inputMessageData = messaging.CmdTorqueBodyMsgPayload()  # Create a structure for the input message
    requestedTorque = [1.0, -0.5, 0.7] # Set up a list as a 3-vector
    inputMessageData.torqueRequestBody = requestedTorque # write torque request to input message
    cmdTorqueInMsg = messaging.CmdTorqueBodyMsg().write(inputMessageData)

    # wheelConfigData message
    rwConfigParams = messaging.RWArrayConfigMsgPayload()
    MAX_EFF_CNT = messaging.MAX_EFF_CNT

    if numWheels == MAX_EFF_CNT:
        rwConfigParams.GsMatrix_B = [
            0.4835867893995201, 0.7025829597277155, 0.5220354411517549,
            0.6274167231454653, 0.4634123147571517, 0.6257773422303058,
            0.4927675437195689, 0.3909468277672152, 0.7773935462269635,
            0.2791305379092009, 0.20278639222840245, 0.9385967301954065,
            0.1742148051521812, 0.9353106472878886, 0.3079662233682429,
            0.7408864742367625, 0.30733781515416325, 0.5971856492492805,
            0.49166240509756476, 0.11024265612126483, 0.863779275153674,
            0.08522980139648922, 0.5635691254043687, 0.8216603445736381,
            0.5169183283391889, 0.6482094982986043, 0.5591242153068406,
            0.5539478507672101, 0.4352935184619988, 0.7096910112262675,
            0.08177103922211226, 0.7185493168899821, 0.6906521384470449,
            0.5424303480563135, 0.8034905566669417, 0.24530031156636306,
            0.6791649825098244, 0.25103926707369056, 0.6897203874901293,
            0.6662787689368599, 0.6695372377111813, 0.32831766535181106,
            0.28428078464167594, 0.5440295499812461, 0.7894404880867942,
            0.8881073966834958, 0.007176386091829566, 0.4595799728433832,
            0.7043700914244455, 0.20398698108861654, 0.6798912308987893,
            0.5913513581668906, 0.7154722881784563, 0.3720255045596441,
            0.5353927164036736, 0.8292977052562882, 0.1600623480977027,
            0.5626385603464779, 0.5530980227747188, 0.6144269099038059,
            0.8047402627946283, 0.5179828986694456, 0.2899772855298006,
            0.6435726414836709, 0.49863310510036174, 0.5806714059015666,
            0.2533767502100278, 0.8066673674024603, 0.533936307831739,
            0.051675625147813466, 0.741898369799065, 0.6685180914942186,
            0.6705007071467579, 0.243658731626882, 0.700756180292173,
            0.6124322825812726, 0.6044312394389204, 0.5094993386086216,
            0.5025822950964116, 0.49662160344788164, 0.7076567103083798,
            0.4875326918964735, 0.8575174427431412, 0.16424283766253403,
            0.3659744927810267, 0.8415919620749859, 0.39722240622155974,
            0.6205921515961875, 0.5508152351685801, 0.5580931446303532,
            0.20125257120061574, 0.7022636474963218, 0.6828785924235018,
            0.4318909377763495, 0.6786025351852008, 0.5941117883924572,
            0.6839787443692367, 0.6598940110591041, 0.31098709204629277,
            0.35743175000357147, 0.8343049491885353, 0.4197353878920623,
            0.8124751056450826, 0.35669421673672336, 0.46114362020262967,
            0.04721328350343224, 0.8901899787392832, 0.45313652204714083]
    else:
        rwConfigParams.GsMatrix_B = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            0.5773502691896258, 0.5773502691896258, 0.5773502691896258
        ]
        rwConfigParams.JsList = [0.1]*numWheels

    rwConfigParams.numRW = numWheels
    rwConfigInMsg = messaging.RWArrayConfigMsg().write(rwConfigParams)

    if RWAvailMsg != "NO":
        rwAvailabilityMessage = messaging.RWAvailabilityMsgPayload()

        avail = [messaging.UNAVAILABLE] * numWheels
        for i in range(numWheels):
            if RWAvailMsg == "ON":
                avail[i] = messaging.AVAILABLE
            elif RWAvailMsg == "OFF":
                avail[i] = messaging.UNAVAILABLE
            else:
                if i < int(numWheels / 2):
                    avail[i] = messaging.AVAILABLE

        rwAvailabilityMessage.wheelAvailability = avail

        rwAvailInMsg = messaging.RWAvailabilityMsg().write(rwAvailabilityMessage)
        module.rwAvailInMsg.subscribeTo(rwAvailInMsg)

    else:
        avail = [rwMotorTorque.AVAILABLE] * numWheels  # this is used purely for the python level solution

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.rwMotorTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.vehControlInMsg.subscribeTo(cmdTorqueInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    module.Reset(0)

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.5))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    # Note that range(3) will provide [0, 1, 2]  Those are the elements you get from the vector (all of them)
    moduleOutput = dataLog.motorTorque

    trueVector = np.array([
        [0.0] * MAX_EFF_CNT,
        [0.0] * MAX_EFF_CNT
    ])

    # set the output truth states
    trueVector[0] = results_rwMotorTorque.computeTorqueU(np.array(controlAxes_B),
                                                                   np.array(rwConfigParams.GsMatrix_B).reshape((
                                                                       3, MAX_EFF_CNT), order='F'),
                                                                   requestedTorque,
                                                                   avail)
    trueVector[1] = trueVector[0]

    # compare the module results to the truth values
    accuracy = 1e-8
    testFailCount, testMessages = unitTestSupport.compareArrayND(trueVector, moduleOutput, accuracy, "rwMotorTorques",
                                                                 MAX_EFF_CNT, testFailCount, testMessages)


    GsMatrix = np.transpose(np.reshape(rwConfigParams.GsMatrix_B,(MAX_EFF_CNT,3),"C"))
    F = np.transpose(moduleOutput[0])
    receivedTorque = -1.0*np.array([np.matmul(GsMatrix,F)])
    receivedTorque = np.append(np.array([]), receivedTorque)

    if numWheels >= numControlAxes and numControlAxes > 0:
        if (len(avail) - np.sum(avail)) > numControlAxes:
            testFailCount, testMessages = unitTestSupport.compareArrayND(np.array([requestedTorque]),
                                                                         np.array([receivedTorque]), accuracy,
                                                                         "CompareTorques",
                                                                         numControlAxes, testFailCount, testMessages)

    snippetName = "LrBReq_LrBRec_"+str(numControlAxes) + "_" + str(numWheels) + "_" + RWAvailMsg
    requestedTex = str(requestedTorque)
    receivedTex = str(receivedTorque[1:4])
    snippetTex = "Requested:\t" + requestedTex + "\n"
    snippetTex += "Received:\t" + receivedTex + "\n"

    unitTestSupport.writeTeXSnippet(snippetName, snippetTex, path)

    #   print out success message if no error were found
    unitTestSupport.writeTeXSnippet('toleranceValue', str(accuracy), path)

    snippentName = "passFail_"+str(numControlAxes) + str(numWheels) + RWAvailMsg
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_rwMotorTorque(False,
                3,      # numControlAxes
                36,      # numWheels
                "NO"    # RWAvailMsg ("NO", "ON", "OFF")
               )
