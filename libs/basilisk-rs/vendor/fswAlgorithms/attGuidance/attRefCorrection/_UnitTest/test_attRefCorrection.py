# 
#  ISC License
# 
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado Boulder
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

import math

import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import attRefCorrection
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport


@pytest.mark.parametrize("accuracy", [1e-12])

def test_attRefCorrection(show_plots, accuracy):
    r"""
    **Validation Test Description**

    Checks the output of the module that the correct orientation adjustment is applied

    **Test Parameters**

    Args:
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    The ``sigma_RN`` variable of the output message is tested
    """
    [testResults, testMessage] = attRefCorrectionTestFunction(show_plots, accuracy)
    assert testResults < 1, testMessage


def attRefCorrectionTestFunction(show_plots, accuracy):
    """Test method"""
    testFailCount = 0
    testMessages = []
    unitTaskName = "unitTask"
    unitProcessName = "TestProcess"

    unitTestSim = SimulationBaseClass.SimBaseClass()
    testProcessRate = macros.sec2nano(0.5)
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # setup module to be tested
    module = attRefCorrection.attRefCorrection()
    module.ModelTag = "attRefCorrectionTag"
    unitTestSim.AddModelToTask(unitTaskName, module)
    module.sigma_BcB = [math.tan(math.pi/4), 0.0, 0.0]

    # Configure blank module input messages
    attRefInMsgData = messaging.AttRefMsgPayload()
    attRefInMsgData.sigma_RN = [math.tan(math.pi/8), 0.0, 0.0]
    attRefInMsg = messaging.AttRefMsg().write(attRefInMsgData)

    # subscribe input messages to module
    module.attRefInMsg.subscribeTo(attRefInMsg)

    # setup output message recorder objects
    attRefOutMsgRec = module.attRefOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, attRefOutMsgRec)

    unitTestSim.InitializeSimulation()
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))
    unitTestSim.ExecuteSimulation()

    # pull module data and make sure it is correct
    trueVector = [
        [-math.tan(math.pi / 8), 0.0, 0.0],
        [-math.tan(math.pi / 8), 0.0, 0.0],
        [-math.tan(math.pi / 8), 0.0, 0.0]
    ]
    # compare the module results to the truth values
    for i in range(0, len(trueVector)):
        # check a vector values
        if not unitTestSupport.isArrayEqual(attRefOutMsgRec.sigma_RN[i], trueVector[i], 3, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed sigma_RN unit test at t=" +
                                str(attRefOutMsgRec.times()[i] * macros.NANO2SEC) +
                                "sec\n")

    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)
    else:
        print(testMessages)

    return [testFailCount, "".join(testMessages)]


if __name__ == "__main__":
    test_attRefCorrection(False, 1e-12)


