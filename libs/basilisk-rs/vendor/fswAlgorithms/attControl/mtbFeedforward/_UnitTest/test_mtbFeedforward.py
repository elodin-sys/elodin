
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
#   Module Name:        mtbFeedforward
#   Author:             Henry Macanas
#   Creation Date:      06 18, 2021
#
# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.
import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import mtbFeedforward  # import the module that is to be tested
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

def test_mtbFeedforward_module():     # update "module" in this function name to reflect the module name
    r"""
    **Validation Test Description**

    This script tests that the torqueRequestBody vector is computed as 
    expected and that the algorithm doesn't fail when given inputs with a 
    value of zero.

    **Description of Variables Being Tested**

    In this file we are checking the values of the output message variable:

    - ``torqueRequestBody``
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = mtbFeedforwardModuleTestFunction()
    assert testResults < 1, testMessage
    
def mtbFeedforwardModuleTestFunction():
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
    module = mtbFeedforward.mtbFeedforward()
    module.ModelTag = "mrpFeedback"           # update python name of test module
    unitTestSim.AddModelToTask(unitTaskName, module)
    
    # Initialize CmdTorqueBodyMsg
    vehControlInMsgContainer = messaging.CmdTorqueBodyMsgPayload()
    vehControlInMsgContainer.torqueRequestBody = [0., 0., 0.]
    vehControlInMsg = messaging.CmdTorqueBodyMsg().write(vehControlInMsgContainer)
    
    # Initialize DipoleRequestBodyMsg
    dipoleRequestMtbInMsgContainer = messaging.MTBCmdMsgPayload()
    dipoleRequestMtbInMsgContainer.mtbDipoleCmds = [1., 2., 3.]
    dipoleRequestMtbInMsg = messaging.MTBCmdMsg().write(dipoleRequestMtbInMsgContainer) 
    
    # Initialize TAMSensorBodyMsg
    tamSensorBodyInMsgContainer = messaging.TAMSensorBodyMsgPayload()
    tamSensorBodyInMsgContainer.tam_B = [ 1E-5, -3E-5, 5E-5]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)

    # Initialize MTBArrayConfigMsg
    mtbArrayConfigParamsInMsgContainer = messaging.MTBArrayConfigMsgPayload()
    mtbArrayConfigParamsInMsgContainer.numMTB = 3
    mtbArrayConfigParamsInMsgContainer.maxMtbDipoles = [1E3, 1E3, 1E3]
    mtbArrayConfigParamsInMsgContainer.GtMatrix_B = [1., 0., 0., 0., 1., 0., 0., 0., 1.]  
    mtbArrayConfigParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbArrayConfigParamsInMsgContainer)

    # Setup logging on the test module output message so that we get all the writes to it
    resultVehControlOutMsg = module.vehControlOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, resultVehControlOutMsg)
    
    # connect the message interfaces
    module.vehControlInMsg.subscribeTo(vehControlInMsg)
    module.dipoleRequestMtbInMsg.subscribeTo(dipoleRequestMtbInMsg)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    module.mtbArrayConfigParamsInMsg.subscribeTo(mtbArrayConfigParamsInMsg)
    
    # Set the simulation time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.0))        # seconds to stop simulation
    unitTestSim.InitializeSimulation()

    '''
        TEST 1: 
            Check that dipoles are non-zero expected value.
    '''
    unitTestSim.ExecuteSimulation()
    m = np.array(dipoleRequestMtbInMsgContainer.mtbDipoleCmds[0:3])
    b = np.array(tamSensorBodyInMsgContainer.tam_B)
    expectedTorque = -np.cross(m, b)    
    testFailCount, testMessages = unitTestSupport.compareVector(expectedTorque,
                                                                resultVehControlOutMsg.torqueRequestBody[0],
                                                                accuracy,
                                                                "torqueRequestBody",
                                                                testFailCount, testMessages)                        
    
    '''
        TEST 2: 
            Check that torqueRequestBody is zero when b field is zero.
    '''
    tamSensorBodyInMsgContainer.tam_B = [0., 0., 0.]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    expectedTorque = [0., 0., 0.]   
    testFailCount, testMessages = unitTestSupport.compareVector(expectedTorque,
                                                                resultVehControlOutMsg.torqueRequestBody[0],
                                                                accuracy,
                                                                "torqueRequestBody",
                                                                testFailCount, testMessages)
    
    '''
        TEST 3: 
            Check that torqueRequestBody is zero when dipoles are zero.
    '''
    tamSensorBodyInMsgContainer.tam_B = [1E-5, -3E-5, 5E-5]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    
    dipoleRequestMtbInMsgContainer.mtbDipoleCmds = [0., 0., 0.]
    dipoleRequestMtbInMsg = messaging.MTBCmdMsg().write(dipoleRequestMtbInMsgContainer) 
    module.dipoleRequestMtbInMsg.subscribeTo(dipoleRequestMtbInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    expectedTorque = [0., 0., 0.]   
    testFailCount, testMessages = unitTestSupport.compareVector(expectedTorque,
                                                                resultVehControlOutMsg.torqueRequestBody[0],
                                                                accuracy,
                                                                "torqueRequestBody",
                                                                testFailCount, testMessages)
    
    '''
        TEST 4: 
            Check that torqueRequestBody is non-zero expected value with 
            non-trivial Gt matrix.
    '''
    dipoleRequestMtbInMsgContainer.mtbDipoleCmds = [7., -3.]
    dipoleRequestMtbInMsg = messaging.MTBCmdMsg().write(dipoleRequestMtbInMsgContainer) 
    module.dipoleRequestMtbInMsg.subscribeTo(dipoleRequestMtbInMsg)
    
    beta = 45. * np.pi / 180.
    Gt = np.array([[np.cos(beta), -np.sin(beta)],[np.sin(beta),  np.cos(beta)], [0., 0.]])
    mtbArrayConfigParamsInMsgContainer.numMTB = 2
    mtbArrayConfigParamsInMsgContainer.GtMatrix_B = [Gt[0, 0], Gt[0, 1], 
                                                     Gt[1, 0], Gt[1, 1],
                                                     Gt[2, 0], Gt[2, 1]]  
    mtbArrayConfigParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbArrayConfigParamsInMsgContainer)
    module.mtbArrayConfigParamsInMsg.subscribeTo(mtbArrayConfigParamsInMsg)

    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    m = Gt @ np.array(dipoleRequestMtbInMsgContainer.mtbDipoleCmds[0:2])
    b = np.array(tamSensorBodyInMsgContainer.tam_B)
    expectedTorque = -np.cross(m, b) 
    testFailCount, testMessages = unitTestSupport.compareVector(expectedTorque,
                                                                resultVehControlOutMsg.torqueRequestBody[0],
                                                                accuracy,
                                                                "torqueRequestBody",
                                                                testFailCount, testMessages)
    
    print("Accuracy used: " + str(accuracy))
    if testFailCount == 0:
        print("PASSED: mtbFeedforward unit test")
    else:
        print("Failed: mtbFeedforward unit test")
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_mtbFeedforward_module()
    