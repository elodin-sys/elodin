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
#   Integrated Unit Test Script
#   Purpose:  Self-check on the AVS C-code support libraries
#   Author:  Hanspeter Schaub
#   Creation Date:  August 11, 2017
#

import os

import pytest
from Basilisk.architecture import avsLibrarySelfCheck
from Basilisk.utilities import unitTestSupport


@pytest.mark.parametrize("testName",
                         ["testRigidBodyKinematics"
                          , "testOrbitalElements"
                          , "testOrbitalAnomalies"
                          , "testLinearAlgebra"
                          , "testOrbitalHill"
                          , "testEnvironment"])
# provide a unique test method name, starting with test_
def test_unitDynamicsModes(testName):
    """AVS Library Self Check"""
    # each test method requires a single assert method to be called
    [testResults, testMessage] = unitAVSLibrarySelfCheck(testName)
    assert testResults < 1, testMessage


def unitAVSLibrarySelfCheck(testName):
    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages

    if testName == "testRigidBodyKinematics":
        errorCount = avsLibrarySelfCheck.testRigidBodyKinematics(1e-10)
        if errorCount:
            testFailCount += errorCount
            testMessages.append("ERROR: Rigid Body Kinematics Library Failed Self Test.\n")
    if testName == "testOrbitalAnomalies":
        errorCount = avsLibrarySelfCheck.testOrbitalAnomalies(1e-10)
        if errorCount:
            testFailCount += errorCount
            testMessages.append("ERROR: Orbital Anomalies Library Failed Self Test.\n")
    if testName == "testOrbitalHill":
        errorCount = avsLibrarySelfCheck.testOrbitalHill(1e-4)
        if errorCount:
            testFailCount += errorCount
            testMessages.append("ERROR: Orbital Hill Library Failed Self Test.\n")
    if testName == "testLinearAlgebra":
        errorCount = avsLibrarySelfCheck.testLinearAlgebra(1e-10)
        if errorCount:
            testFailCount += errorCount
            testMessages.append("ERROR: Linear Algebra Library Failed Self Test.\n")

    if testFailCount == 0:
        print("PASSED ")
        passFailText = "PASSED"
        colorText = 'ForestGreen'  # color to write auto-documented "PASSED" message in in LATEX
        snippetContent = ""
    else:
        print(testFailCount)
        print(testMessages)
        passFailText = 'FAILED'
        colorText = 'Red'  # color to write auto-documented "FAILED" message in in LATEX
        snippetContent = ""
        for message in testMessages:
            snippetContent += message

    fileName = os.path.basename(os.path.splitext(__file__)[0])
    path = os.path.dirname(os.path.abspath(__file__))

    snippetMsgName = fileName + 'Msg-' + testName
    unitTestSupport.writeTeXSnippet(snippetMsgName, snippetContent, path + "/../_Documentation/")

    snippetPassFailName = fileName + 'TestMsg-' + testName
    snippetContent = r'\textcolor{' + colorText + '}{' + passFailText + '}'
    unitTestSupport.writeTeXSnippet(snippetPassFailName, snippetContent, path + "/../_Documentation/")

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unit test scrip can be run as a
# stand-along python script
#
if __name__ == "__main__":
    unitAVSLibrarySelfCheck(
        "testOrbitalHill"
    )
