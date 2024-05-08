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
#   Module Name:        dipoleMapping
#   Author:             Henry Macanas
#   Creation Date:      06 18, 2021
#
# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.
import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import dipoleMapping  # import the module that is to be tested
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

def test_dipoleMapping_module():     # update "module" in this function name to reflect the module name
    r"""
    **Validation Test Description**

    This script tests the mapping of a 3x1 requested Body frame dipole, 
    ``dipole_B``, mapped correctly to individual torque bar requests and that the
    algorithm doesn't fail when the inputs are given zero values.

    **Description of Variables Being Tested**

    In this file we are checking the values of the output message variable:

    - ``mtbDipoleCmds[MAX_EFF_CNT]``
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = dipoleMappingModuleTestFunction()
    assert testResults < 1, testMessage
    
def dipoleMappingModuleTestFunction():
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
    module = dipoleMapping.dipoleMapping()
    module.steeringMatrix = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    module.ModelTag = "dipoleMapping"           # update python name of test module
    unitTestSim.AddModelToTask(unitTaskName, module)
    
    # Initialize DipoleRequestBodyMsg
    dipoleRequestBodyInMsgContainer = messaging.DipoleRequestBodyMsgPayload()
    dipoleRequestBodyInMsgContainer.dipole_B = [1., 2., 3.]
    dipoleRequestBodyInMsg = messaging.DipoleRequestBodyMsg().write(dipoleRequestBodyInMsgContainer)    

    # Initialize MTBArrayConfigMsg
    mtbArrayConfigParamsInMsgContainer = messaging.MTBArrayConfigMsgPayload()
    mtbArrayConfigParamsInMsgContainer.numMTB = 3
    mtbArrayConfigParamsInMsgContainer.maxMtbDipoles = [1E3, 1E3, 1E3]
    mtbArrayConfigParamsInMsgContainer.GtMatrix_B = [1., 0., 0., 0., 1., 0., 0., 0., 1.]  
    mtbArrayConfigParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbArrayConfigParamsInMsgContainer)

    # Setup logging on the test module output message so that we get all the writes to it
    resultDipoleRequestMtbOutMsg = module.dipoleRequestMtbOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, resultDipoleRequestMtbOutMsg)
    
    # connect the message interfaces
    module.dipoleRequestBodyInMsg.subscribeTo(dipoleRequestBodyInMsg)
    module.mtbArrayConfigParamsInMsg.subscribeTo(mtbArrayConfigParamsInMsg)
    
    # Set the simulation time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.0))        # seconds to stop simulation
    unitTestSim.InitializeSimulation()                        
    
    '''
        TEST 1: 
            Check that dipoles is non-zero expected value with trivial 
            steeringMatrix.
    '''
    unitTestSim.ExecuteSimulation()
    expectedDipole = [0.] * messaging.MAX_EFF_CNT
    expectedDipole[0:3] = [1., 2., 3.]
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                                resultDipoleRequestMtbOutMsg.mtbDipoleCmds[0],
                                                                accuracy,
                                                                "dipoles",
                                                                testFailCount, testMessages)
    '''
        TEST 2: 
            Check that dipoles is non-zero with non-trivial steeringMatrix.
    '''
    beta = 45. * np.pi / 180.
    Gt = np.array([[np.cos(beta), -np.sin(beta)],[np.sin(beta),  np.cos(beta)], [0., 0.]])
    GtInverse = np.linalg.pinv(Gt)
    mtbArrayConfigParamsInMsgContainer.numMTB = 2
    mtbArrayConfigParamsInMsgContainer.GtMatrix_B = [Gt[0, 0], Gt[0, 1], 
                                                     Gt[1, 0], Gt[1, 1],
                                                     Gt[2, 0], Gt[2, 1]]  
    mtbArrayConfigParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbArrayConfigParamsInMsgContainer)
    module.mtbArrayConfigParamsInMsg.subscribeTo(mtbArrayConfigParamsInMsg)
    
    module.steeringMatrix = [GtInverse[0, 0], GtInverse[0, 1], GtInverse[0, 2],
                                   GtInverse[1, 0], GtInverse[1, 1], GtInverse[1, 2]]
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    
    expectedDipole = [0.] * messaging.MAX_EFF_CNT
    expectedDipole[0:2] = GtInverse @ np.array(dipoleRequestBodyInMsgContainer.dipole_B)
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                            resultDipoleRequestMtbOutMsg.mtbDipoleCmds[0],
                                                            accuracy,
                                                            "dipoles",
                                                            testFailCount, testMessages)
    
    '''
        TEST 3: 
            Check that dipoles is zero with zero input dipole.
    '''
    dipoleRequestBodyInMsgContainer.dipole_B = [0., 0., 0.]
    dipoleRequestBodyInMsg = messaging.DipoleRequestBodyMsg().write(dipoleRequestBodyInMsgContainer)
    module.dipoleRequestBodyInMsg.subscribeTo(dipoleRequestBodyInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    
    unitTestSim.ExecuteSimulation()
    expectedDipole = [0.] * messaging.MAX_EFF_CNT
    testFailCount, testMessages = unitTestSupport.compareVector(expectedDipole,
                                                                resultDipoleRequestMtbOutMsg.mtbDipoleCmds[0],
                                                                accuracy,
                                                                "dipoles",
                                                                testFailCount, testMessages)

    print("Accuracy used: " + str(accuracy))
    if testFailCount == 0:
        print("PASSED: dipoleMapping unit test")
    else:
        print("Failed: dipoleMapping unit test")
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_dipoleMapping_module()
    