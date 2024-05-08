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
#   Module Name:        torque2Dipole
#   Author:             Henry Macanas
#   Creation Date:      06 18, 2021
#
# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.
import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import torque2Dipole  # import the module that is to be tested
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions

accuracy = 1E-12

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

def test_torque2Dipole_module():     # update "module" in this function name to reflect the module name
    r"""
    **Validation Test Description**

    This script tests that the 3x1 Body frame dipole vector, 
    dipole_B, is computed correctly and that the algorithm doesn't fail when
    the inputs are given zero values.

    **Description of Variables Being Tested**

    In this file we are checking the values of the variable:

    - ``dipole_B[3]``
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = torque2DipoleModuleTestFunction()
    assert testResults < 1, testMessage
    
def torque2DipoleModuleTestFunction():
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.01)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Initialize module under test's config message and add module to runtime call list
    module = torque2Dipole.torque2Dipole()
    module.ModelTag = "mtbMomentumManagement"           # update python name of test module
    unitTestSim.AddModelToTask(unitTaskName, module)
    
    # Initialize TAMSensorBodyMsg
    tamSensorBodyInMsgContainer = messaging.TAMSensorBodyMsgPayload()
    tamSensorBodyInMsgContainer.tam_B = [1E-5, 0.0, 0.0]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    
    # Initialize CmdTorqueBodyMsg
    tauRequestInMsgContainer = messaging.CmdTorqueBodyMsgPayload()
    tauRequestInMsgContainer.torqueRequestBody = [10.* 1E-3, 20. * 1E-3, 30 * 1E-3]
    tauRequestInMsg = messaging.CmdTorqueBodyMsg().write(tauRequestInMsgContainer)
    
    # Setup logging on the test module output message so that we get all the writes to it
    resultDipoleRequestOutMsg = module.dipoleRequestOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, resultDipoleRequestOutMsg)
    
    # connect the message interfaces
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    module.tauRequestInMsg.subscribeTo(tauRequestInMsg)
    
    # Set the simulation time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.0))        # seconds to stop simulation
    unitTestSim.InitializeSimulation()
    
    '''
        TEST 1: 
            Check that dipole_B is non-zero expected value.
    '''
    unitTestSim.ExecuteSimulation()
    b = np.array(tamSensorBodyInMsgContainer.tam_B)
    tau = np.array(tauRequestInMsgContainer.torqueRequestBody)
    expectedDipole = 1 / np.dot(b, b) * np.cross(b, tau)
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                                resultDipoleRequestOutMsg.dipole_B[0],
                                                                accuracy,
                                                                "dipole_B",
                                                                testFailCount, testMessages)
    
    '''
        TEST 2: 
            Check that dipole_B is zero when tam_B is zero.
    '''
    tamSensorBodyInMsgContainer.tam_B = [0., 0., 0.]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    
    expectedDipole = [0., 0., 0.]
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                                resultDipoleRequestOutMsg.dipole_B[0],
                                                                accuracy,
                                                                "dipole_B",
                                                                testFailCount, testMessages)
    
    '''
        TEST 3: 
            Check that dipole_B is zero when torqueRequestBody is zero.
    '''
    tamSensorBodyInMsgContainer.tam_B = [1E-5, 0.0, 0.0]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    
    tauRequestInMsgContainer.torqueRequestBody = [0., 0., 0.]
    tauRequestInMsg = messaging.CmdTorqueBodyMsg().write(tauRequestInMsgContainer)
    module.tauRequestInMsg.subscribeTo(tauRequestInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    
    expectedDipole = [0., 0., 0.]
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                                resultDipoleRequestOutMsg.dipole_B[0],
                                                                accuracy,
                                                                "dipole_B",
                                                                testFailCount, testMessages)
    
    
    print("Accuracy used: " + str(accuracy))
    if testFailCount == 0:
        print("PASSED: torque2Dipole unit test")
    else:
        print("Failed: torque2Dipole unit test")
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_torque2Dipole_module()
    