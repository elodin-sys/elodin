#
#  ISC License
#
#  Copyright (c) 2022, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
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
#   Module Name:        torqueScheduler
#   Author:             Riccardo Calaon
#   Creation Date:      January 25, 2023
#

import pytest
from Basilisk.architecture import bskLogging
from Basilisk.architecture import messaging  # import the message definitions
from Basilisk.fswAlgorithms import torqueScheduler  # import the module that is to be tested
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
# of the multiple test runs for this test.  Note that the order in that you add the parametrize method
# matters for the documentation in that it impacts the order in which the test arguments are shown.
# The first parametrize arguments are shown last in the pytest argument list
@pytest.mark.parametrize("lockFlag", [0, 1, 2, 3])
@pytest.mark.parametrize("tSwitch", [3, 6])
@pytest.mark.parametrize("accuracy", [1e-12])


def test_torqueScheduler(lockFlag, tSwitch, accuracy):
    r"""
    **Validation Test Description**

    This unit test verifies the correctness of the output motor torque :ref:`torqueScheduler`.
    The inputs provided are the lock flag and the time at which thr control is switched from 
    one degree of freedom to the other.

    **Test Parameters**

    Args:
        lockFlag (int): flag to determine which torque to use first;
        tSwitch (double): time at which torque is to be switched from one d.o.f. to the other;

    **Description of Variables Being Tested**

    This unit test checks the correctness of the output motor torque msg and the output effector lock msg:

    - ``motorTorqueOutMsg``
    - ``effectorLockOutMsg``.

    The test checks that the output of ``motorTorqueOutMsg`` always matches the torques contained in the input msgs
    and that the flags contained in ``effectorLockOutMsg`` are consistent with the schedule logic that the user is requesting.
    """
    # each test method requires a single assert method to be called
    [testResults, testMessage] = torqueSchedulerTestFunction(lockFlag, tSwitch, accuracy)
    assert testResults < 1, testMessage


def torqueSchedulerTestFunction(lockFlag, tSwitch, accuracy):

    testFailCount = 0                        # zero unit test result counter
    testMessages = []                        # create empty array to store test log messages
    unitTaskName = "unitTask"                # arbitrary name (don't change)
    unitProcessName = "TestProcess"          # arbitrary name (don't change)
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(1)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))

    # Construct algorithm and associated C container
    scheduler = torqueScheduler.torqueScheduler()
    scheduler.ModelTag = "torqueScheduler"
    scheduler.lockFlag = lockFlag
    scheduler.tSwitch = tSwitch
    unitTestSim.AddModelToTask(unitTaskName, scheduler)

    # Create input array motor torque msg #1
    motorTorque1InMsgData = messaging.ArrayMotorTorqueMsgPayload()
    motorTorque1InMsgData.motorTorque = [1]
    motorTorque1InMsg = messaging.ArrayMotorTorqueMsg().write(motorTorque1InMsgData)
    scheduler.motorTorque1InMsg.subscribeTo(motorTorque1InMsg)

    # Create input array motor torque msg #2
    motorTorque2InMsgData = messaging.ArrayMotorTorqueMsgPayload()
    motorTorque2InMsgData.motorTorque = [3]
    motorTorque2InMsg = messaging.ArrayMotorTorqueMsg().write(motorTorque2InMsgData)
    scheduler.motorTorque2InMsg.subscribeTo(motorTorque2InMsg)

    # Setup logging on the test module output messages so that we get all the writes to it
    torqueLog = scheduler.motorTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, torqueLog)
    lockLog = scheduler.effectorLockOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, lockLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(10))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # compare the module results to the truth values
    time = torqueLog.times() * macros.NANO2SEC

    for i in range(len(time)):
        if not unitTestSupport.isDoubleEqual(torqueLog.motorTorque[i][0], motorTorque1InMsgData.motorTorque[0], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at passing motor torque #1 value")
        if not unitTestSupport.isDoubleEqual(torqueLog.motorTorque[i][1], motorTorque2InMsgData.motorTorque[0], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at passing motor torque #2 value")

        if lockFlag == 0:
            if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 0, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
            if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 0, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")
        elif lockFlag == 1:
            if time[i] > tSwitch:
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 1, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 0, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")
            else:
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 0, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 1, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")
        elif lockFlag == 2:
            if time[i] > tSwitch:
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 0, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 1, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")
            else:
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 1, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
                if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 0, accuracy):
                    testFailCount += 1
                    testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")
        else:
            if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][0], 1, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #1")
            if not unitTestSupport.isDoubleEqual(lockLog.effectorLockFlag[i][1], 1, accuracy):
                testFailCount += 1
                testMessages.append("FAILED: " + scheduler.ModelTag + " module failed at outputting effector flag #2")

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_torqueScheduler( 
                 1,
                 5,
                 1e-12
               )
