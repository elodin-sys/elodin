
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


import inspect
import math
import os

import numpy
import pytest

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
from Basilisk import __path__
bskPath = __path__[0]

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.fswAlgorithms import chebyPosEphem
from Basilisk.topLevelModules import pyswice
from Basilisk.utilities.pyswice_spk_utilities import spkRead
import matplotlib.pyplot as plt
from Basilisk.architecture import messaging

orbitPosAccuracy = 1.0
orbitVelAccuracy = 0.01

# uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed
# @pytest.mark.skipif(conditionstring)
# uncomment this line if this test has an expected failure, adjust message as needed
# @pytest.mark.xfail() # need to update how the RW states are defined
# provide a unique test method name, starting with test_
@pytest.mark.parametrize("function", ["sineCosine"
                                      , "earthOrbitFit"
                                      ])
def test_chebyPosFitAllTest(show_plots, function):
    """Module Unit Test"""
    [testResults, testMessage] = eval(function + '(show_plots)')
    assert testResults < 1, testMessage


def sineCosine(show_plots):
    """Module Unit Test"""
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    __tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    orbitRadius = 70000.0
    numCurvePoints = 365*3+1
    curveDurationDays = 365.0*3
    degChebCoeff =21

    angleSpace = numpy.linspace(-3*math.pi, 3*math.pi, numCurvePoints)

    cosineValues = numpy.cos(angleSpace)*orbitRadius
    sineValues = numpy.sin(angleSpace)*orbitRadius
    oopValues = numpy.sin(angleSpace) + orbitRadius

    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/naif0012.tls')
    et = pyswice.new_doubleArray(1)

    timeStringMid = '2019 APR 1 12:12:12.0 (UTC)'
    pyswice.str2et_c(timeStringMid, et)

    fitTimes = numpy.linspace(-1, 1, numCurvePoints)

    chebCosCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, cosineValues, degChebCoeff)
    chebSinCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, sineValues, degChebCoeff)
    cheboopCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, oopValues, degChebCoeff)

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    TotalSim = SimulationBaseClass.SimBaseClass()

    FSWUnitTestProc = TotalSim.CreateNewProcess(unitProcessName)
    # create the dynamics task and specify the integration update time
    FSWUnitTestProc.addTask(TotalSim.CreateNewTask(unitTaskName, macros.sec2nano(8640.0)))

    chebyFitModel = chebyPosEphem.chebyPosEphem()
    chebyFitModel.ModelTag = "chebyFitModel"
    TotalSim.AddModelToTask(unitTaskName, chebyFitModel)

    totalList = numpy.array(chebCosCoeff).tolist()
    totalList.extend(numpy.array(chebSinCoeff).tolist())
    totalList.extend(numpy.array(cheboopCoeff).tolist())

    chebyFitModel.ephArray[0].posChebyCoeff = totalList
    chebyFitModel.ephArray[0].nChebCoeff = degChebCoeff+1
    chebyFitModel.ephArray[0].ephemTimeMid = pyswice.doubleArray_getitem(et, 0)
    chebyFitModel.ephArray[0].ephemTimeRad = curveDurationDays/2.0*86400.0

    clockCorrData = messaging.TDBVehicleClockCorrelationMsgPayload()
    clockCorrData.vehicleClockTime = 0.0
    clockCorrData.ephemerisTime = chebyFitModel.ephArray[0].ephemTimeMid  - \
        chebyFitModel.ephArray[0].ephemTimeRad
    clockInMsg = messaging.TDBVehicleClockCorrelationMsg().write(clockCorrData)
    chebyFitModel.clockCorrInMsg.subscribeTo(clockInMsg)

    xFitData = numpy.polynomial.chebyshev.chebval(fitTimes, chebCosCoeff)

    dataLog = chebyFitModel.posFitOutMsg.recorder()
    TotalSim.AddModelToTask(unitTaskName, dataLog)

    TotalSim.InitializeSimulation()
    TotalSim.ConfigureStopTime(int(curveDurationDays*86400.0*1.0E9))
    TotalSim.ExecuteSimulation()

    posChebData = dataLog.r_BdyZero_N

    angleSpaceFine = numpy.linspace(-3*math.pi, 3*math.pi, numCurvePoints*10-9)

    cosineValuesFine = numpy.cos(angleSpaceFine)*orbitRadius
    sineValuesFine = numpy.sin(angleSpaceFine)*orbitRadius
    oopValuesFine = numpy.sin(angleSpaceFine) + orbitRadius

    maxErrVec = [max(abs(posChebData[:,0] - cosineValuesFine)),
        max(abs(posChebData[:,1] - sineValuesFine)),
        max(abs(posChebData[:,2] - oopValuesFine))]

    print("Sine Wave error: " +  str(max(maxErrVec)))
    assert max(maxErrVec) < orbitPosAccuracy

    if testFailCount == 0:
        print("PASSED: " + " Sine and Cosine curve fit")
    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

def earthOrbitFit(show_plots):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    #__tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    numCurvePoints = 365*3+1
    curveDurationSeconds = 3*5950.0
    degChebCoeff =23
    integFrame = "j2000"
    zeroBase = "Earth"

    dateSpice = "2015 February 10, 00:00:00.0 TDB"
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/naif0012.tls')
    et = pyswice.new_doubleArray(1)
    pyswice.str2et_c(dateSpice, et)
    etStart = pyswice.doubleArray_getitem(et, 0)
    etEnd = etStart + curveDurationSeconds

    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/de430.bsp')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/naif0012.tls')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/de-403-masses.tpc')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/pck00010.tpc')
    pyswice.furnsh_c(path + '/hst_edited.bsp')

    hubblePosList = []
    hubbleVelList = []
    timeHistory = numpy.linspace(etStart, etEnd, numCurvePoints)

    for timeVal in timeHistory:
        stringCurrent = pyswice.et2utc_c(timeVal, 'C', 4, 1024, "Yo")
        stateOut = spkRead('HUBBLE SPACE TELESCOPE', stringCurrent, integFrame, zeroBase)
        hubblePosList.append(stateOut[0:3].tolist())
        hubbleVelList.append(stateOut[3:6].tolist())

    hubblePosList = numpy.array(hubblePosList)
    hubbleVelList = numpy.array(hubbleVelList)

    fitTimes = numpy.linspace(-1, 1, numCurvePoints)
    chebCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, hubblePosList, degChebCoeff)

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    TotalSim = SimulationBaseClass.SimBaseClass()

    FSWUnitTestProc = TotalSim.CreateNewProcess(unitProcessName)
    # create the dynamics task and specify the integration update time
    FSWUnitTestProc.addTask(TotalSim.CreateNewTask(unitTaskName, macros.sec2nano(curveDurationSeconds/(numCurvePoints-1))))

    chebyFitModel = chebyPosEphem.chebyPosEphem()
    chebyFitModel.ModelTag = "chebyFitModel"
    TotalSim.AddModelToTask(unitTaskName, chebyFitModel)

    totalList = chebCoeff[:,0].tolist()
    totalList.extend(chebCoeff[:,1].tolist())
    totalList.extend(chebCoeff[:,2].tolist())

    chebyFitModel.ephArray[0].posChebyCoeff = totalList
    chebyFitModel.ephArray[0].nChebCoeff = degChebCoeff+1
    chebyFitModel.ephArray[0].ephemTimeMid = etStart + curveDurationSeconds/2.0
    chebyFitModel.ephArray[0].ephemTimeRad = curveDurationSeconds/2.0

    clockCorrData = messaging.TDBVehicleClockCorrelationMsgPayload()
    clockCorrData.vehicleClockTime = 0.0
    clockCorrData.ephemerisTime = chebyFitModel.ephArray[0].ephemTimeMid  - \
        chebyFitModel.ephArray[0].ephemTimeRad
    clockInMsg = messaging.TDBVehicleClockCorrelationMsg().write(clockCorrData)
    chebyFitModel.clockCorrInMsg.subscribeTo(clockInMsg)

    dataLog = chebyFitModel.posFitOutMsg.recorder()
    TotalSim.AddModelToTask(unitTaskName, dataLog)

    TotalSim.InitializeSimulation()
    TotalSim.ConfigureStopTime(int(curveDurationSeconds*1.0E9))
    TotalSim.ExecuteSimulation()

    posChebData = dataLog.r_BdyZero_N
    velChebData = dataLog.v_BdyZero_N

    maxErrVec = [abs(max(posChebData[:,0] - hubblePosList[:,0])),
        abs(max(posChebData[:,1] - hubblePosList[:,1])),
        abs(max(posChebData[:,2] - hubblePosList[:,2]))]
    maxVelErrVec = [abs(max(velChebData[:,0] - hubbleVelList[:,0])),
             abs(max(velChebData[:,1] - hubbleVelList[:,1])),
             abs(max(velChebData[:,2] - hubbleVelList[:,2]))]
    print("Hubble Orbit Accuracy: " + str(max(maxErrVec)))
    print("Hubble Velocity Accuracy: " + str(max(maxVelErrVec)))
    assert (max(maxErrVec)) < orbitPosAccuracy
    assert (max(maxVelErrVec)) < orbitVelAccuracy
    plt.figure()
    plt.plot(dataLog.times()*1.0E-9, velChebData[:,0], dataLog.times()*1.0E-9, hubbleVelList[:,0])

    if(show_plots):
        plt.show()
        plt.close('all')

    if testFailCount == 0:
        print("PASSED: " + " Orbit curve fit")
    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]

if __name__ == "__main__":
    test_chebyPosFitAllTest(True)
