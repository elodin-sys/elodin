
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

import matplotlib.pyplot as plt
import numpy
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import oeStateEphem
from Basilisk.topLevelModules import pyswice
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import unitTestSupport
from Basilisk.utilities.pyswice_spk_utilities import spkRead

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
splitPath = path.split('fswAlgorithms')
from Basilisk import __path__
bskPath = __path__[0]

orbitPosAccuracy = 10000.0
orbitVelAccuracy = 1.0
unitTestSupport.writeTeXSnippet("tolerancePosValue", str(orbitPosAccuracy), path)
unitTestSupport.writeTeXSnippet("toleranceVelValue", str(orbitVelAccuracy), path)


@pytest.mark.parametrize('validChebyCurveTime, anomFlag', [
    (True, 0),
    (True, 1),
    (True, -1),
    (False, -1)
])
def test_chebyPosFitAllTest(show_plots, validChebyCurveTime, anomFlag):
    """Module Unit Test"""
    [testResults, testMessage] = chebyPosFitAllTest(show_plots, validChebyCurveTime, anomFlag)
    assert testResults < 1, testMessage


def chebyPosFitAllTest(show_plots, validChebyCurveTime, anomFlag):
    # The __tracebackhide__ setting influences pytest showing of tracebacks:
    # the mrp_steering_tracking() function will not be shown unless the
    # --fulltrace command line option is specified.
    #__tracebackhide__ = True

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty list to store test log messages

    numCurvePoints = 4*8640+1
    curveDurationSeconds = 4*86400
    logPeriod = curveDurationSeconds // (numCurvePoints - 1)
    degChebCoeff = 14
    integFrame = "j2000"
    zeroBase = "Earth"
    centralBodyMu = 3.98574405096E14

    dateSpice = "2015 April 10, 00:00:00.0 TDB"
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/naif0012.tls')
    et = pyswice.new_doubleArray(1)
    pyswice.str2et_c(dateSpice, et)
    etStart = pyswice.doubleArray_getitem(et, 0)
    etEnd = etStart + curveDurationSeconds

    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/de430.bsp')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/naif0012.tls')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/de-403-masses.tpc')
    pyswice.furnsh_c(bskPath + '/supportData/EphemerisData/pck00010.tpc')
    pyswice.furnsh_c(path + '/TDRSS.bsp')

    tdrssPosList = []
    tdrssVelList = []
    timeHistory = numpy.linspace(etStart, etEnd, numCurvePoints)
    position = numpy.array(3)
    velocity = numpy.array(3)
    rpArray = []
    eccArray = []
    incArray = []
    OmegaArray = []
    omegaArray = []
    anomArray = []
    anomPrev = 0.0
    anomCount = 0

    for timeVal in timeHistory:
        stringCurrent = pyswice.et2utc_c(timeVal, 'C', 4, 1024, "Yo")
        stateOut = spkRead('-221', stringCurrent, integFrame, zeroBase)
        position = stateOut[0:3]*1000.0
        velocity = stateOut[3:6]*1000.0
        orbEl = orbitalMotion.rv2elem(centralBodyMu, position, velocity)
        tdrssPosList.append(position)
        tdrssVelList.append(velocity)
        rpArray.append(orbEl.rPeriap)
        eccArray.append(orbEl.e)
        incArray.append(orbEl.i)
        OmegaArray.append(orbEl.Omega)
        omegaArray.append(orbEl.omega)
        if anomFlag == 1:
            currentAnom = orbitalMotion.E2M(orbitalMotion.f2E(orbEl.f, orbEl.e), orbEl.e)
        else:
            currentAnom = orbEl.f
        if currentAnom < anomPrev:
            anomCount += 1
        anomArray.append(2*math.pi*anomCount + currentAnom)
        anomPrev = currentAnom

    tdrssPosList = numpy.array(tdrssPosList)
    tdrssVelList = numpy.array(tdrssVelList)

    fitTimes = numpy.linspace(-1, 1, numCurvePoints)
    chebRpCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, rpArray, degChebCoeff)
    chebEccCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, eccArray, degChebCoeff)
    chebIncCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, incArray, degChebCoeff)
    chebOmegaCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, OmegaArray, degChebCoeff)
    chebomegaCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, omegaArray, degChebCoeff)
    chebAnomCoeff = numpy.polynomial.chebyshev.chebfit(fitTimes, anomArray, degChebCoeff)

    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    sim = SimulationBaseClass.SimBaseClass()

    FSWUnitTestProc = sim.CreateNewProcess(unitProcessName)
    # create the dynamics task and specify the integration update time
    FSWUnitTestProc.addTask(sim.CreateNewTask(unitTaskName, macros.sec2nano(logPeriod)))

    oeStateModel = oeStateEphem.oeStateEphem()
    oeStateModel.ModelTag = "oeStateModel"
    sim.AddModelToTask(unitTaskName, oeStateModel)

    oeStateModel.muCentral = centralBodyMu

    oeStateModel.ephArray[0].rPeriapCoeff = chebRpCoeff.tolist()
    oeStateModel.ephArray[0].eccCoeff = chebEccCoeff.tolist()
    oeStateModel.ephArray[0].incCoeff = chebIncCoeff.tolist()
    oeStateModel.ephArray[0].argPerCoeff = chebomegaCoeff.tolist()
    oeStateModel.ephArray[0].anomCoeff = chebAnomCoeff.tolist()
    oeStateModel.ephArray[0].RAANCoeff = chebOmegaCoeff.tolist()
    oeStateModel.ephArray[0].nChebCoeff = degChebCoeff + 1
    oeStateModel.ephArray[0].ephemTimeMid = etStart + curveDurationSeconds/2.0
    oeStateModel.ephArray[0].ephemTimeRad = curveDurationSeconds/2.0

    if not (anomFlag == -1):
        oeStateModel.ephArray[0].anomalyFlag = anomFlag

    clockCorrData = messaging.TDBVehicleClockCorrelationMsgPayload()
    clockCorrData.vehicleClockTime = 0.0
    clockCorrData.ephemerisTime = oeStateModel.ephArray[0].ephemTimeMid - oeStateModel.ephArray[0].ephemTimeRad

    clockInMsg = messaging.TDBVehicleClockCorrelationMsg().write(clockCorrData)
    oeStateModel.clockCorrInMsg.subscribeTo(clockInMsg)

    dataLog = oeStateModel.stateFitOutMsg.recorder()
    sim.AddModelToTask(unitTaskName, dataLog)

    if not validChebyCurveTime:
        sim.InitializeSimulation()
        # increase the run time by one logging period so that the sim time is outside the
        # valid chebychev curve duration
        sim.ConfigureStopTime(int((curveDurationSeconds + logPeriod) * 1.0E9))
        sim.ExecuteSimulation()
    else:
        sim.InitializeSimulation()
        sim.ConfigureStopTime(int(curveDurationSeconds*1.0E9))
        sim.ExecuteSimulation()

    posChebData = dataLog.r_BdyZero_N
    velChebData = dataLog.v_BdyZero_N

    if not validChebyCurveTime:
        lastLogidx = (curveDurationSeconds + logPeriod) // logPeriod - 1
        secondLastPos = posChebData[lastLogidx + 1, 0:] - tdrssPosList[lastLogidx, :]
        lastPos = posChebData[lastLogidx, 0:] - tdrssPosList[lastLogidx, :]
        if not numpy.array_equal(secondLastPos, lastPos):
            testFailCount += 1
            testMessages.append("FAILED: Expected Chebychev position to rail high or low "
                                + str(secondLastPos)
                                + " != "
                                + str(lastPos))

        secondLastVel = velChebData[lastLogidx + 1, 0:] - tdrssVelList[lastLogidx, :]
        lastVel = velChebData[lastLogidx, 0:] - tdrssVelList[lastLogidx, :]
        if not numpy.array_equal(secondLastVel, lastVel):
            testFailCount += 1
            testMessages.append("FAILED: Expected Chebychev velocity to rail high or low "
                                + str(secondLastVel)
                                + " != "
                                + str(lastVel))
    else:
        maxErrVec = [abs(max(posChebData[:, 0] - tdrssPosList[:, 0])),
                     abs(max(posChebData[:, 1] - tdrssPosList[:, 1])),
                     abs(max(posChebData[:,2] - tdrssPosList[:, 2]))]
        maxVelErrVec = [abs(max(velChebData[:, 0] - tdrssVelList[:, 0])),
                        abs(max(velChebData[:, 1] - tdrssVelList[:, 1])),
                        abs(max(velChebData[:, 2] - tdrssVelList[:, 2]))]

        if max(maxErrVec) >= orbitPosAccuracy:
            testFailCount += 1
            testMessages.append("FAILED: maxErrVec >= orbitPosAccuracy, TDRSS Orbit Accuracy: "
                                + str(max(maxErrVec)))
        if max(maxVelErrVec) >= orbitVelAccuracy:
            testFailCount += 1
            testMessages.append("FAILED: maxVelErrVec >= orbitVelAccuracy, TDRSS Velocity Accuracy: "
                                + str(max(maxVelErrVec)))

        plt.close("all")
        # plot the fitted and actual position coordinates
        plt.figure(1)
        fig = plt.gcf()
        ax = fig.gca()
        ax.ticklabel_format(useOffset=False, style='plain')
        for idx in range(0, 3):
            plt.plot(dataLog.times()*macros.NANO2HOUR,
                     posChebData[:, idx]/1000,
                     color=unitTestSupport.getLineColor(idx, 3),
                     linewidth=0.5,
                     label='$r_{fit,' + str(idx) + '}$')
            plt.plot(dataLog.times()*macros.NANO2HOUR,
                     tdrssPosList[:, idx]/1000,
                     color=unitTestSupport.getLineColor(idx, 3),
                     linestyle='dashed', linewidth=2,
                     label='$r_{true,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [h]')
        plt.ylabel('Inertial Position [km]')

        # plot the fitted and actual velocity coordinates
        plt.figure(2)
        for idx in range(0, 3):
            plt.plot(dataLog.times()*macros.NANO2HOUR,
                     velChebData[:, idx]/1000,
                     color=unitTestSupport.getLineColor(idx, 3),
                     linewidth=0.5,
                     label='$v_{fit,' + str(idx) + '}$')
            plt.plot(dataLog.times()*macros.NANO2HOUR,
                     tdrssVelList[:, idx]/1000,
                     color=unitTestSupport.getLineColor(idx, 3),
                     linestyle='dashed', linewidth=2,
                     label='$v_{true,' + str(idx) + '}$')
        plt.legend(loc='lower right')
        plt.xlabel('Time [h]')
        plt.ylabel('Velocity [km/s]')

        # plot the difference in position coordinates
        plt.figure(3)
        arrayLength = posChebData[:, 0].size
        for idx in range(0,3):
            plt.plot(dataLog.times() * macros.NANO2HOUR,
                     posChebData[:, idx] - tdrssPosList[:, idx],
                     color=unitTestSupport.getLineColor(idx, 3),
                     linewidth=0.5,
                     label=r'$\Delta r_{' + str(idx) + '}$')
        plt.plot(dataLog.times() * macros.NANO2HOUR,
                 orbitPosAccuracy*numpy.ones(arrayLength),
                 color='r',
                 linewidth=1)
        plt.plot(dataLog.times() * macros.NANO2HOUR,
                 -orbitPosAccuracy * numpy.ones(arrayLength),
                 color='r',
                 linewidth=1)
        plt.legend(loc='lower right')
        plt.xlabel('Time [h]')
        plt.ylabel('Position Difference [m]')

        # plot the difference in velocity coordinates
        plt.figure(4)
        arrayLength = velChebData[:, 0].size
        for idx in range(0,3):
            plt.plot(dataLog.times() * macros.NANO2HOUR,
                     velChebData[:, idx] - tdrssVelList[:, idx],
                     color=unitTestSupport.getLineColor(idx, 3),
                     linewidth=0.5,
                     label=r'$\Delta v_{' + str(idx) + '}$')
        plt.plot(dataLog.times() * macros.NANO2HOUR,
                 orbitVelAccuracy*numpy.ones(arrayLength),
                 color='r',
                 linewidth=1)
        plt.plot(dataLog.times() * macros.NANO2HOUR,
                 -orbitVelAccuracy * numpy.ones(arrayLength),
                 color='r',
                 linewidth=1)
        plt.legend(loc='lower right')
        plt.xlabel('Time [h]')
        plt.ylabel('Velocity Difference [m/s]')

    if show_plots:
        plt.show()
        plt.close('all')

    snippentName = "passFail" + str(validChebyCurveTime)
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + oeStateModel.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + oeStateModel.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)

    # return fail count and join into a single string all messages in the list
    # testMessage
    return [testFailCount, ''.join(testMessages)]


if __name__ == "__main__":
    chebyPosFitAllTest(True,        # showPlots
                       True,        # validChebyCurveTime
                       1)           # anomFlag
