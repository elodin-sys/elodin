
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
#   Module Name:        centerRadiusCNN
#   Author:             Thibaud Teil
#   Creation Date:      January 10, 2020
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
    reasonErr = "python Pillow package not installed---can't test CenterRadiusCNN module"

# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros

try:
    from Basilisk.fswAlgorithms import centerRadiusCNN
except ImportError:
    importErr = True
    reasonErr = "CenterRadiusCNN not built---check opNav option"

# Uncomment this line is this test is to be skipped in the global unit test run, adjust message as needed.
# @pytest.mark.skipif(conditionstring)
# Uncomment this line if this test has an expected failure, adjust message as needed.
# @pytest.mark.xfail(conditionstring)
# Provide a unique test method name, starting with 'test_'.

@pytest.mark.skipif(importErr, reason= reasonErr)
@pytest.mark.parametrize("image, saveImage", [
                    ("mars.jpg", False),
                   ("mars2.jpg", False),
                   ("mars3.jpg", False)
])

# update "module" in this function name to reflect the module name
def test_module(show_plots, image, saveImage):
    """
    Unit test for CNN module for radius and center finding. Each parameter reads a different image.
    They are both images of Mars, and were not in the training data.

    """
    # each test method requires a single assert method to be called
    [testResults, testMessage] = cnnTest(show_plots, image, saveImage)
    assert testResults < 1, testMessage


def cnnTest(show_plots, image, saveImage):

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
    module = centerRadiusCNN.CenterRadiusCNN()
    module.ModelTag = "cnn"

    # Add test module to runtime call list
    unitTestSim.AddModelToTask(unitTaskName, module)

    module.pathToNetwork = path + "/../CAD.onnx"
    module.filename = imagePath
    module.pixelNoise = [5,5,5]

    circles = []
    if image == "mars.jpg":
        circles = [(254.81655883789062, 263.2418518066406, 120.9570541381836)]
    if image == "mars2.jpg":
        circles = [(269.21127319, 231.63162231, 144.85394287)]

    # Setup logging on the test module output message so that we get all the writes to it
    dataLog = module.opnavCirclesOutMsg.recorder()
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

    centers = dataLog.circlesCenters[:, :10*2]
    radii = dataLog.circlesRadii[:, :10]

    # Output image:
    output_image = Image.new("RGB", input_image.size)
    output_image.paste(input_image)
    draw_result = ImageDraw.Draw(output_image)

    imageProcCircles = []
    for j in range(len(radii[-1, 0:])):
        if radii[-1,j] > 0:
            imageProcCircles.append((centers[-1, 2*j], centers[-1, 2*j+1], radii[-1, j]))
    for x, y, r in imageProcCircles:
        draw_result.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0, 0))

    # Save output image
    if saveImage:
        output_image.save("result_"+ image)

    if show_plots:
        print(imageProcCircles[0])
        output_image.show()


    #   print out success message if no error were found
    for testCircle, refCircle in zip(imageProcCircles, circles):
        for i in range(3):
            if np.abs((testCircle[i] - refCircle[i])/refCircle[i])>1:
                testFailCount+=1
                testMessages.append("Test failed processing " + image)


    # each test method requires a single assert method to be called
    # this check below just makes sure no sub-test failures were found
    return [testFailCount, ''.join(testMessages)]


#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    if importErr:
        print(reasonErr)
        exit(1)
    cnnTest(True, "mars.jpg", True) # mars images
