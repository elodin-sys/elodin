
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
#   Module Name:        navAggregate()
#   Author:             Hanspeter Schaub
#   Creation Date:      Feb. 21, 2019
#

import inspect
import os

import numpy as np
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
bskName = 'Basilisk'
splitPath = path.split(bskName)







# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import unitTestSupport
from Basilisk.fswAlgorithms import navAggregate
from Basilisk.utilities import macros
from Basilisk.architecture import messaging

# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.
# The following 'parametrize' function decorator provides the parameters and expected results for each
#   of the multiple test runs for this test.
@pytest.mark.parametrize("numAttNav, numTransNav", [
      (0, 0)
    , (1, 1)
    , (0, 1)
    , (1, 0)
    , (2, 2)
    , (1, 2)
    , (0, 2)
    , (2, 1)
    , (2, 0)
    , (3, 3)
    , (3, 2)
    , (3, 1)
    , (3, 0)
    , (2, 3)
    , (1, 3)
    , (0, 3)
    , (11, 11)
    , (3, 11)
    , (2, 11)
    , (1, 11)
    , (0, 11)
    , (11, 3)
    , (11, 2)
    , (11, 1)
    , (11, 0)
])

# update "module" in this function name to reflect the module name
def test_module(show_plots, numAttNav, numTransNav):
    """Module Unit Test"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = navAggregateTestFunction(show_plots, numAttNav, numTransNav)
    assert testResults < 1, testMessage


def navAggregateTestFunction(show_plots, numAttNav, numTransNav):
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

    # Construct an instance of the module being tested
    module = navAggregate.navAggregate()
    module.ModelTag = "navAggregate"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    # Create input messages
    navAtt1Msg = messaging.NavAttMsgPayload()
    navAtt1Msg.timeTag = 11.11
    navAtt1Msg.sigma_BN = [0.1, 0.01, -0.1]
    navAtt1Msg.omega_BN_B = [1., 1., -1.]
    navAtt1Msg.vehSunPntBdy = [-0.1, 0.1, 0.1]
    navAtt1InMsg = messaging.NavAttMsg().write(navAtt1Msg)
    navAtt2Msg = messaging.NavAttMsgPayload()
    navAtt2Msg.timeTag = 22.22
    navAtt2Msg.sigma_BN = [0.2, 0.02, -0.2]
    navAtt2Msg.omega_BN_B = [2., 2., -2.]
    navAtt2Msg.vehSunPntBdy = [-0.2, 0.2, 0.2]
    navAtt2InMsg = messaging.NavAttMsg().write(navAtt2Msg)

    navTrans1Msg = messaging.NavTransMsgPayload()
    navTrans1Msg.timeTag = 11.1
    navTrans1Msg.r_BN_N = [1000.0, 100.0, -1000.0]
    navTrans1Msg.v_BN_N = [1., 1., -1.]
    navTrans1Msg.vehAccumDV = [-10.1, 10.1, 10.1]
    navTrans1InMsg = messaging.NavTransMsg().write(navTrans1Msg)
    navTrans2Msg = messaging.NavTransMsgPayload()
    navTrans2Msg.timeTag = 22.2
    navTrans2Msg.r_BN_N = [2000.0, 200.0, -2000.0]
    navTrans2Msg.v_BN_N = [2., 2., -2.]
    navTrans2Msg.vehAccumDV = [-20.2, 20.2, 20.2]
    navTrans2InMsg = messaging.NavTransMsg().write(navTrans2Msg)

    # create input navigation message containers
    navAtt1 = navAggregate.AggregateAttInput()
    navAtt2 = navAggregate.AggregateAttInput()
    navTrans1 = navAggregate.AggregateTransInput()
    navTrans2 = navAggregate.AggregateTransInput()

    module.attMsgCount = numAttNav
    if numAttNav == 3:       # here the index asks to read from an empty (zero) message
        module.attMsgCount = 2

    module.transMsgCount = numTransNav
    if numTransNav == 3:     # here the index asks to read from an empty (zero) message
        module.transMsgCount = 2

    if numAttNav <= navAggregate.MAX_AGG_NAV_MSG:
        module.attMsgs = [navAtt1, navAtt2]
        module.attMsgs[0].navAttInMsg.subscribeTo(navAtt1InMsg)
        module.attMsgs[1].navAttInMsg.subscribeTo(navAtt2InMsg)
    else:
        module.attMsgs = [navAtt1] * navAggregate.MAX_AGG_NAV_MSG
        for i in range(navAggregate.MAX_AGG_NAV_MSG):
            module.attMsgs[i].navAttInMsg.subscribeTo(navAtt1InMsg)
    if numTransNav <= navAggregate.MAX_AGG_NAV_MSG:
        module.transMsgs = [navTrans1, navTrans2]
        module.transMsgs[0].navTransInMsg.subscribeTo(navTrans1InMsg)
        module.transMsgs[1].navTransInMsg.subscribeTo(navTrans2InMsg)
    else:
        module.transMsgs = [navTrans1] * navAggregate.MAX_AGG_NAV_MSG
        for i in range(navAggregate.MAX_AGG_NAV_MSG):
            module.transMsgs[i].navTransInMsg.subscribeTo(navTrans1InMsg)

    if numAttNav > 1:       # always read from the last message counter
        module.attTimeIdx = numAttNav - 1
        module.attIdx = numAttNav - 1
        module.rateIdx = numAttNav - 1
        module.sunIdx = numAttNav - 1
    if numTransNav > 1:     # always read from the last message counter
        module.transTimeIdx = numTransNav-1
        module.posIdx = numTransNav-1
        module.velIdx = numTransNav-1
        module.dvIdx = numTransNav-1

    # write TeX snippets for the message values
    unitTestSupport.writeTeXSnippet("navAtt1Msg.timeTag", str(navAtt1Msg.timeTag), path)
    unitTestSupport.writeTeXSnippet("navAtt1Msg.sigma_BN", str(navAtt1Msg.sigma_BN), path)
    unitTestSupport.writeTeXSnippet("navAtt1Msg.omega_BN_B", str(navAtt1Msg.omega_BN_B), path)
    unitTestSupport.writeTeXSnippet("navAtt1Msg.vehSunPntBdy", str(navAtt1Msg.vehSunPntBdy), path)
    unitTestSupport.writeTeXSnippet("navAtt2Msg.timeTag", str(navAtt2Msg.timeTag), path)
    unitTestSupport.writeTeXSnippet("navAtt2Msg.sigma_BN", str(navAtt2Msg.sigma_BN), path)
    unitTestSupport.writeTeXSnippet("navAtt2Msg.omega_BN_B", str(navAtt2Msg.omega_BN_B), path)
    unitTestSupport.writeTeXSnippet("navAtt2Msg.vehSunPntBdy", str(navAtt2Msg.vehSunPntBdy), path)
    unitTestSupport.writeTeXSnippet("navTrans1Msg.timeTag", str(navTrans1Msg.timeTag), path)
    unitTestSupport.writeTeXSnippet("navTrans1Msg.r_BN_N", str(navTrans1Msg.r_BN_N), path)
    unitTestSupport.writeTeXSnippet("navTrans1Msg.v_BN_N", str(navTrans1Msg.v_BN_N), path)
    unitTestSupport.writeTeXSnippet("navTrans1Msg.vehAccumDV", str(navTrans1Msg.vehAccumDV), path)
    unitTestSupport.writeTeXSnippet("navTrans2Msg.timeTag", str(navTrans2Msg.timeTag), path)
    unitTestSupport.writeTeXSnippet("navTrans2Msg.r_BN_N", str(navTrans2Msg.r_BN_N), path)
    unitTestSupport.writeTeXSnippet("navTrans2Msg.v_BN_N", str(navTrans2Msg.v_BN_N), path)
    unitTestSupport.writeTeXSnippet("navTrans2Msg.vehAccumDV", str(navTrans2Msg.vehAccumDV), path)

    # Setup logging on the test module output message so that we get all the writes to it
    dataAttLog = module.navAttOutMsg.recorder()
    dataTransLog = module.navTransOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataAttLog)
    unitTestSim.AddModelToTask(unitTaskName, dataTransLog)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    # This pulls the actual data log from the simulation run.
    attTimeTag = np.transpose([dataAttLog.timeTag])
    attSigma = dataAttLog.sigma_BN
    attOmega = dataAttLog.omega_BN_B
    attSunVector = dataAttLog.vehSunPntBdy

    transTimeTag = np.transpose([dataTransLog.timeTag])
    transPos = dataTransLog.r_BN_N
    transVel = dataTransLog.v_BN_N
    transAccum = dataTransLog.vehAccumDV

    # set the filtered output truth states
    if numAttNav == 0 or numAttNav == 3:
        trueAttTimeTag = [[0.0]]*3
        trueAttSigma = [[0., 0., 0.]]*3
        trueAttOmega = [[0., 0., 0.]]*3
        trueAttSunVector = [[0., 0., 0.]]*3

    if numTransNav == 0 or numTransNav == 3:
        trueTransTimeTag = [[0.0]]*3
        trueTransPos = [[0.0, 0.0, 0.0]]*3
        trueTransVel = [[0.0, 0.0, 0.0]]*3
        trueTransAccum = [[0.0, 0.0, 0.0]]*3

    if numAttNav == 1 or numAttNav == 11:
        trueAttTimeTag = [[navAtt1Msg.timeTag]]*3
        trueAttSigma = [navAtt1Msg.sigma_BN]*3
        trueAttOmega = [navAtt1Msg.omega_BN_B]*3
        trueAttSunVector = [navAtt1Msg.vehSunPntBdy]*3

    if numTransNav == 1 or numTransNav == 11:
        trueTransTimeTag = [[navTrans1Msg.timeTag]]*3
        trueTransPos = [navTrans1Msg.r_BN_N]*3
        trueTransVel = [navTrans1Msg.v_BN_N]*3
        trueTransAccum = [navTrans1Msg.vehAccumDV]*3

    if numAttNav == 2:
        trueAttTimeTag = [[navAtt2Msg.timeTag]] * 3
        trueAttSigma = [navAtt2Msg.sigma_BN] * 3
        trueAttOmega = [navAtt2Msg.omega_BN_B] * 3
        trueAttSunVector = [navAtt2Msg.vehSunPntBdy] * 3

    if numTransNav == 2:
        trueTransTimeTag = [[navTrans2Msg.timeTag]]*3
        trueTransPos = [navTrans2Msg.r_BN_N]*3
        trueTransVel = [navTrans2Msg.v_BN_N]*3
        trueTransAccum = [navTrans2Msg.vehAccumDV]*3

    # compare the module results to the truth values
    accuracy = 1e-12
    unitTestSupport.writeTeXSnippet("toleranceValue", str(accuracy), path)

    # check if the module output matches the truth data
    testFailCount, testMessages = unitTestSupport.compareArrayND(trueAttTimeTag, attTimeTag,
                                                               accuracy, "attTimeTag", 1,
                                                               testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueAttSigma, attSigma,
                                                                 accuracy, "sigma_BN",
                                                                 testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueAttOmega, attOmega,
                                                               accuracy, "omega_BN_B",
                                                               testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueAttSunVector, attSunVector,
                                                               accuracy, "vehSunPntBdy",
                                                               testFailCount, testMessages)

    testFailCount, testMessages = unitTestSupport.compareArrayND(trueTransTimeTag, transTimeTag,
                                                                 accuracy, "transTimeTag", 1,
                                                                 testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueTransPos, transPos,
                                                               accuracy, "sigma_BN",
                                                               testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueTransVel, transVel,
                                                               accuracy, "omega_BN_B",
                                                               testFailCount, testMessages)
    testFailCount, testMessages = unitTestSupport.compareArray(trueTransAccum, transAccum,
                                                               accuracy, "vehSunPntBdy",
                                                               testFailCount, testMessages)

    if numAttNav == 11:
        if module.attMsgCount != navAggregate.MAX_AGG_NAV_MSG:
            testFailCount += 1
            testMessages.append("FAILED numAttNav too large test")
        if module.attTimeIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED attTimeIdx too large test")
        if module.attIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED attIdx too large test")
        if module.rateIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED rateIdx too large test")
        if module.sunIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED sunIdx too large test")

    if numTransNav == 11:
        if module.transMsgCount != navAggregate.MAX_AGG_NAV_MSG:
            testFailCount += 1
            testMessages.append("FAILED numTransNav too large test")
        if module.posIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED posIdx too large test")
        if module.velIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED velIdx too large test")
        if module.dvIdx != navAggregate.MAX_AGG_NAV_MSG-1:
            testFailCount += 1
            testMessages.append("FAILED dvIdx too large test")

    #   print out success message if no error were found
    snippentName = "passFail" + str(numAttNav) + str(numTransNav)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + module.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)

    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_module(
                 False,
                 2,             # numAttNav
                 2              # numTransNav
               )
