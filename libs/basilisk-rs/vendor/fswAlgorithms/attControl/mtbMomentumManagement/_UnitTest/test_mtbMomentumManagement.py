#
#  ISC License
#
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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
#   Module Name:        mtbMomentumManagement
#   Author:             Henry Macanas
#   Creation Date:      02 23, 2021
#
# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.
import numpy as np
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import mtbMomentumManagement  # import the module that is to be tested
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions

# CONSTANTS
MAX_EFF_CNT = 36

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_

def test_mtbMomentumManagement():     # update "module" in this function name to reflect the module name
    r"""
    **Validation Test Description**

    This script tests that the module returns expected non-zero and zero 
    outputs.

    **Description of Variables Being Tested**

    The variables being checked are:
    variables

    - ``mtbDipoleCmds[MAX_EFF_CNT]``
    - ``motorTorque[MAX_EFF_CNT]``
    """
    # each test method requires a single assert method to be called
    # pass on the testPlotFixture so that the main test function may set the DataStore attributes
    [testResults, testMessage] = mtbMomentumManagementModuleTestFunction()
    assert testResults < 1, testMessage

def mtbMomentumManagementModuleTestFunction():
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

    # Construct algorithm and associated C++ container
    module = mtbMomentumManagement.mtbMomentumManagement()
    module.cGain = 0.005
    module.wheelSpeedBiases = [0., 0., 0, 0.]
    module.ModelTag = "mtbMomentumManagement"           # update python name of test module
    unitTestSim.AddModelToTask(unitTaskName, module)

    # wheelConfigData message (array is ordered c11, c22, c33, c44, ...)
    rwConfigParams = messaging.RWArrayConfigMsgPayload()
    beta = 45. * np.pi / 180.
    rwConfigParams.GsMatrix_B = [0., np.cos(beta), np.sin(beta), 0., np.sin(beta), -np.cos(beta), np.cos(beta), -np.sin(beta), 0., -np.cos(beta), -np.sin(beta), 0.]
    rwConfigParams.JsList = [0.002, 0.002, 0.002, 0.002]
    rwConfigParams.numRW = 4
    rwParamsInMsg = messaging.RWArrayConfigMsg().write(rwConfigParams)
    
    # mtbConfigData message (array is ordered c11, c22, c33, c44, ...)
    mtbConfigParams = messaging.MTBArrayConfigMsgPayload()
    mtbConfigParams.numMTB = 3
    # row major toque bar alignments
    mtbConfigParams.GtMatrix_B = [
        1., 0., 0.,
        0., 1., 0.,
        0., 0., 1.
    ]
    mtbConfigParams.maxMtbDipoles = [10.]*mtbConfigParams.numMTB
    mtbParamsInMsg = messaging.MTBArrayConfigMsg().write(mtbConfigParams)
    
    # TAMSensorBodyMsg message (leads to non-invertible matrix)
    tamSensorBodyInMsgContainer = messaging.TAMSensorBodyMsgPayload()
    tamSensorBodyInMsgContainer.tam_B = [ 5E3 * 0.03782347,  5E3 * 0.49170516, 5E3 * -0.8699399]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    
    # rwSpeeds message
    rwSpeedsInMsgContainer = messaging.RWSpeedMsgPayload()
    rwSpeedsInMsgContainer.wheelSpeeds = [100., 200., 300., 400.]
    rwSpeedsInMsg = messaging.RWSpeedMsg().write(rwSpeedsInMsgContainer)

    # attControl message
    rwMotorTorqueInMsgContainer = messaging.ArrayMotorTorqueMsgPayload()
    rwMotorTorqueInMsgContainer.motorTorque = [0., 0., 0., 0.]
    rwMotorTorqueInMsg = messaging.ArrayMotorTorqueMsg().write(rwMotorTorqueInMsgContainer)
    
    # Setup logging on the test module output message so that we get all the writes to it
    resultMtbCmdOutMsg = module.mtbCmdOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, resultMtbCmdOutMsg)
    resultRwMotorTorqueOutMsg = module.rwMotorTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, resultRwMotorTorqueOutMsg)

    # connect the message interfaces
    module.rwParamsInMsg.subscribeTo(rwParamsInMsg)
    module.mtbParamsInMsg.subscribeTo(mtbParamsInMsg)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    module.rwSpeedsInMsg.subscribeTo(rwSpeedsInMsg)
    module.rwMotorTorqueInMsg.subscribeTo(rwMotorTorqueInMsg)
    
    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(0.0))        # seconds to stop simulation
    accuracy = 1E-8
    
    
    '''
        TEST 0: 
            Check that mtbDipoleCmds and are non-zero.
    '''
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()
    
    testFailCount, testMessages = unitTestSupport.compareVector([0., 0., 0.],
                                                                resultMtbCmdOutMsg.mtbDipoleCmds[0][0:3],
                                                                accuracy,
                                                                "tauMtbRequestOutMsg",
                                                                testFailCount, testMessages, ExpectedResult=0)
    
    testFailCount, testMessages = unitTestSupport.compareVector([0., 0., 0., 0.],
                                                                resultRwMotorTorqueOutMsg.motorTorque[0][0:4],
                                                                accuracy,
                                                                "rwMotorTorqueOutMsg",
                                                                testFailCount, testMessages, ExpectedResult=0)  
    '''

        TEST 1: 
            Check that the mtbDipoleCmds are zero and that the resulting
            torque on the body is zero when the b field is zero.
    '''
    tamSensorBodyInMsgContainer.tam_B = [0., 0., 0.]
    tamSensorBodyInMsg = messaging.TAMSensorBodyMsg().write(tamSensorBodyInMsgContainer)
    module.tamSensorBodyInMsg.subscribeTo(tamSensorBodyInMsg)
    
    unitTestSim.InitializeSimulation()
    unitTestSim.ExecuteSimulation()

    testFailCount, testMessages = unitTestSupport.compareVector([0., 0., 0.],
                                                                resultMtbCmdOutMsg.mtbDipoleCmds[0][0:3],
                                                                accuracy,
                                                                "tauMtbRequestOutMsg",
                                                                testFailCount, testMessages, ExpectedResult=1)
    
    Gs = np.array(rwConfigParams.GsMatrix_B[0:12]).reshape(4, 3).T
    tauBody = Gs @ np.array(resultRwMotorTorqueOutMsg.motorTorque[0][0:4])
    testFailCount, testMessages = unitTestSupport.compareVector([0., 0., 0.],
                                                                tauBody,
                                                                accuracy,
                                                                "rwMotorTorqueOutMsg",
                                                                testFailCount, testMessages, ExpectedResult=1)  


    # reset the module to test this functionality
    module.Reset(0)     # this module reset function needs a time input (in NanoSeconds)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    print("fail count", testFailCount)
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_mtbMomentumManagement()
