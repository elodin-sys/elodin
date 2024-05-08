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
#   Module Name:        lowPassFilterTorqueCommand
#   Author:             Hanspeter Schaub
#   Creation Date:      December 9, 2015
#
import math

import matplotlib.pyplot as plt
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import lowPassFilterTorqueCommand  # import the module that is to be tested
#   Import all of the modules that we are going to call in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport  # general support file with common unit test functions


# import packages as needed e.g. 'numpy', 'ctypes, 'math' etc.

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail(conditionstring)
# provide a unique test method name, starting with test_
def test_lowPassFilterControlTorque(show_plots):     # update "subModule" in this function name to reflect the module name
    """Module Unit Test"""
    [testResults, testMessage] = subModuleTestFunction(show_plots)
    assert testResults < 1, testMessage

def subModuleTestFunction(show_plots):
    #   zero all unit test result gather variables
    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    #   Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()
                                                        # this create a fresh and consistent simulation environment for each test run

    #   Create test thread
    testProcessRate = macros.sec2nano(0.5)         # process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    #   Construct algorithm and associated C++ container
    module = lowPassFilterTorqueCommand.lowPassFilterTorqueCommand()
    module.ModelTag = "lowPassFilterTorqueCommand"      # python name of test module.

    #   Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    #   Initialize the test module configuration data
    module.wc = 0.1*math.pi*2                 #   [rad/s] continous time critical filter frequency
    module.h = 0.5                            #   [s]     filter time step
    module.reset = 1                          #           flag to initialize module states on first run


    #   Create input message and size it because the regular creator of that message
    #   is not part of the test.
    inputMessageData = messaging.CmdTorqueBodyMsgPayload()
    inputMessageData.torqueRequestBody = [1.0, -0.5, 0.7]
    inMsg = messaging.CmdTorqueBodyMsg().write(inputMessageData)

    # setup msg connection
    module.cmdTorqueInMsg.subscribeTo(inMsg)

    #   Setup logging on the test module output message so that we get all the writes to it
    outLog = module.cmdTorqueOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, outLog)

    #   Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    #   Step the simulation to 3*process rate so 4 total steps including zero
    unitTestSim.ConfigureStopTime(macros.sec2nano(1.0))    # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    module.Reset(1)     # this module reset function needs a time input (in NanoSeconds)

    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))        # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    LrF = outLog.torqueRequestBody

    #   set the filtered output truth states
    LrFtrue = [
               [0.2734574719946391,-0.1367287359973196,0.1914202303962474],
               [0.4721359549995794,-0.2360679774997897,0.3304951684997055],
               [0.6164843223022588,-0.3082421611511294,0.4315390256115811],
               [0.2734574719946391,-0.1367287359973196,0.1914202303962474],
               [0.4721359549995794,-0.2360679774997897,0.3304951684997055],
               ]

    #   compare the module and truth results
    for i in range(0,len(LrFtrue)):
        if not unitTestSupport.isArrayEqual(LrF[i], LrFtrue[i], 3, 1e-12):
            testFailCount += 1
            testMessages.append("FAILED: " + module.ModelTag + " Module failed LrFtrue unit test at t=" + str(LrF[i,0]*unitTestSupport.NANO2SEC) + "sec\n")



    # If the argument provided at commandline "--show_plots" evaluates as true,
    # plot all figures
    if show_plots:
        plt.show()
        plt.close('all')

    #   print out success message if no error were found
    if testFailCount == 0:
        print("PASSED: " + module.ModelTag)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    test_lowPassFilterControlTorque(False)
