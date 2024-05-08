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
#   Module Name:        rwMotorTorque
#   Author:             Hanspeter Schaub
#   Creation Date:      July 4, 2016
#

from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import rwMotorTorque
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.

# update "module" in this function name to reflect the module name
def test_rwMotorTorque(show_plots):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = rwMotorTorqueTest(show_plots)
    assert testResults < 1, testMessage


def rwMotorTorqueTest(show_plots):
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
    controlAxes_B = [
             1,0,0
            ,0,1,0
            ,0,0,1
    ]
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
    rwConfigParams.GsMatrix_B = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.5773502691896258, 0.5773502691896258, 0.5773502691896258
    ]
    rwConfigParams.JsList = [0.1, 0.1, 0.1, 0.1]
    rwConfigParams.numRW = 4
    rwConfigInMsg = messaging.RWArrayConfigMsg().write(rwConfigParams)

    # wheelAvailability message
    rwAvailabilityMessage = messaging.RWAvailabilityMsgPayload()
    avail = [messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE, messaging.AVAILABLE]
    rwAvailabilityMessage.wheelAvailability = avail
    rwAvailInMsg = messaging.RWAvailabilityMsg().write(rwAvailabilityMessage)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.rwMotorTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # connect messages
    module.vehControlInMsg.subscribeTo(cmdTorqueInMsg)
    module.rwParamsInMsg.subscribeTo(rwConfigInMsg)
    module.rwAvailInMsg.subscribeTo(rwAvailInMsg)

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
    # print('\n', moduleOutput)

    # set the output truth states
    ans = [0]*messaging.MAX_EFF_CNT
    ans[0:4] = [-0.8, 0.7000000000000001, -0.5, -0.3464101615137755]
    trueVector = [
                   ans,
                   ans
    ]

    # compare the module results to the truth values
    accuracy = 1e-12
    for i in range(0,len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(moduleOutput[i], trueVector[i], rwConfigParams.numRW, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed motorTorque unit test at t=" +
                                str(dataLog.times()[i]*macros.NANO2SEC) +
                                "sec\n")

    #   print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
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
    test_rwMotorTorque(False)
