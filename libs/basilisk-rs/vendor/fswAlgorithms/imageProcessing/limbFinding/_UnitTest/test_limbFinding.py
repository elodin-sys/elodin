
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
#   Module Name:        LimbFinding
#   Author:             Thibaud Teil
#   Creation Date:      September 16, 2019
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
importErr = False
reasonErr = ""
try:
    from PIL import Image, ImageDraw
except ImportError:
    importErr = True
    reasonErr = "python Pillow package not installed---can't test Limb Finding module"

# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.architecture import messaging
try:
    from Basilisk.fswAlgorithms import limbFinding
except ImportError:
    importErr = True
    reasonErr = "Limb Finding not built---check OpenCV option"

# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.

@pytest.mark.skipif(importErr, reason= reasonErr)
@pytest.mark.parametrize("image,         blur,    cannyLow,  cannyHigh, saveImage", [
                        ("MarsBright.jpg",    1,    100,       200,   False), #Mars image
                        ("MarsDark.jpg",      1,    100,       200,   False),  # Mars image
                        ("moons.jpg",         3,    200,       300,   False) # Moon images
    ])

# update "module" in this function name to reflect the module name
def test_module(show_plots, image, blur, cannyLow, cannyHigh, saveImage):
    """
    Unit test for Limb Finding. The unit test specifically runs on 3 images:

        1. A full Mars: This image has an easy to detect, full mars disk

        2. A crescent Mars: This image only contains a slim Mars crescent

        3. Moons: This image contains several Moon crescents

    The Limb Finding module uses the Canny transform in order to find the planet limb. It then outputs all the points
    on the limb.
    """
    # each test method requires a single assert method to be called
    [testResults, testMessage] = limbFindingTest(show_plots, image, blur, cannyLow, cannyHigh, saveImage)
    assert testResults < 1, testMessage


def limbFindingTest(show_plots, image, blur, cannyLow, cannyHigh, saveImage):

    # Truth values from python
    imagePath = path + '/' + image
    input_image = Image.open(imagePath)
    input_image.load()
    #################################################

    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # # Create test thread
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))


    # Construct algorithm and associated C++ container
    module = limbFinding.LimbFinding()
    module.ModelTag = "limbFind"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    module.filename = imagePath
    module.cannyThreshHigh = cannyHigh
    module.cannyThreshLow = cannyLow
    module.blurrSize = blur

    reference = []
    refPoints = 0
    if image == "MarsBright.jpg":
        reference = [253.0, 110.0]
        refPoints = 2*475.0
    if image == "MarsDark.jpg":
        reference = [187.0, 128.0]
        refPoints = 2*192.0
    if image == "moons.jpg":
        reference = [213.0, 66.0]
        refPoints = 2*270.0
    # Create input message and size it because the regular creator of that message
    # is not part of the test.
    inputMessageData = messaging.CameraImageMsgPayload()
    inputMessageData.timeTag = int(1E9)
    inputMessageData.cameraID = 1
    imageInMsg = messaging.CameraImageMsg().write(inputMessageData)
    module.imageInMsg.subscribeTo(imageInMsg)

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.opnavLimbOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)


    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(macros.sec2nano(2.0))        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    valid = dataLog.valid
    points = dataLog.limbPoints[:, :2*1000]
    numPoints = dataLog.numLimbPoints

    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    imageProcLimb = []
    for j in range(int(len(points[-1,1:])/2)):
        if points[-1,2*j]>1E-2:
            imageProcLimb.append((points[-1,2*j], points[-1,2*j+1]))

    draw_result.point(imageProcLimb, fill=128)

    # Save output image
    if saveImage:
        output_image.save("result_"+ image)

    if show_plots:
        output_image.show()


    #   print out success message if no error were found
    for i in range(2):
        if np.abs((reference[i] - imageProcLimb[0][i])/reference[i])>1:
            print(np.abs((reference[i] - imageProcLimb[0][i])/reference[i]))
            testFailCount += 1
            testMessages.append("Limb Test failed processing " + image)
    if valid[-1] != 1:
        testFailCount += 1
        testMessages.append("Validity test failed processing " + image)
    if np.abs(numPoints[-1]-refPoints)>10:
        testFailCount += 1
        testMessages.append("NumPoints test failed processing " + image)

    if testFailCount:
        print(testMessages)
    else:
        print("Passed")

    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    limbFindingTest(True, "MarsBright.jpg",     1,    100,       200,  True) # Moon images
