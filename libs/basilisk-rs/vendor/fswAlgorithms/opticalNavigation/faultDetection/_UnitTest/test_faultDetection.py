#
#   Unit Test Script
#   Module Name:        pixelLineConverter.py
#   Creation Date:      May 16, 2019
#

import inspect
import os
import pytest

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import faultDetection
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, macros

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

@pytest.mark.parametrize("r_c1, r_c2, valid1, valid2, faultMode", [
                      ([10, 10, 1], [10, 10, 1], 1, 1, 1), # No merge, all valid, no fault
                      ([10, 10, 1], [10, 10, 1], 1, 0, 1),  #No merge, one valid, no fault
                      ([10, 10, 1], [10, 10, 1], 1, 0, 2),  #No merge, one valid, no fault, mode 2
                      ([10, 10, 1], [10, 10, 1], 0, 1, 1),  # No merge, other valid, no fault
                      ([10, 10, 1], [10, 10, 1], 0, 1, 2),  # No merge, other valid, no fault, mode 2
                      ([10, 10, 1], [10, 10, 1], 0, 1, 0),  # Merge, other valid, no fault
                      ([10, 10, 1], [10, 10, 1], 0, 0, 1),  # No merge, none valid, no fault
                      ([10, 10, 1], [100, 10, 1], 1, 1, 1),  # No merge, all valid, fault
                      ([10, 10, 1], [100, 10, 1], 1, 1, 0),  # merge, all valid, fault
                      ([10, 10, 1], [11, 9, 0.5], 1, 1, 0),  # merge, all valid, no fault
                      ([10, 10, 1], [11, 9, 0.5], 1, 1, 2),  # merge, all valid, no fault, mode 2
                      ([10, 10, 1], [11, 9, 0.5], 1, 1, 0),  # merge, all valid, no fault

])

def test_faultdetection(show_plots, r_c1, r_c2, valid1, valid2, faultMode):
    """
    **Validation Test Description**

    This module tests the fault detection scenario. The logic behind the fault detection is explained in the doxygen documentation.
    In order to properly test the proper functioning of the fault detection, all the possible combinations are run (8).
    The expected results are computed in python and are tested with the output.

    **Test Parameters**

    absolute accuracy value of 1E-10 is used in this test

    - case 1: ([10, 10, 1], [10, 10, 1], 1, 1, 0)
        No measurement merge, all are valid, no faults
    - case 2: ([10, 10, 1], [10, 10, 1], 1, 0, 0)
        No measurement merge, one valid, no faults
    - case 3: ([10, 10, 1], [10, 10, 1], 1, 0, 2)
        No measurement merge, one valid, no faults, mode 2
    - case 4: ([10, 10, 1], [10, 10, 1], 0, 1, 0)
        No measurement merge, other valid, no faults
    - case 5: ([10, 10, 1], [10, 10, 1], 0, 1, 0)
        No measurement merge, other valid, no faults, mode 2
    - case 6: ([10, 10, 1], [10, 10, 1], 0, 1, 1)
        Merge on, other valid, no faults
    - case 7: ([10, 10, 1], [10, 10, 1], 0, 0, 0)
        No merge, none valid, no faults
    - case 8: ([10, 10, 1], [100, 10, 1], 1, 1, 0)
        No merge, all measurements valid, fault
    - case 9: ([10, 10, 1], [100, 10, 1], 1, 1, 1)
        Merge, all measurements valid, fault
    - case 10: ([10, 10, 1], [100, 10, 1], 1, 1, 2)
        Merge, all measurements valid, fault, mode 2
    - case 11: ([10, 10, 1], [10, 10, 1], 1, 1, 1)
        Merge, all measurements valid, no fault

    **Description of Variables Being Tested**

    The time, detection of a fault, measurement, and measurement covariances are tested on the output.
    These are ``r_BN_N``, ``covar_N``, ``time``, ``faultDetected``

    """
    [testResults, testMessage] = faultdetection(show_plots, r_c1, r_c2, valid1, valid2, faultMode)
    assert testResults < 1, testMessage

def faultdetection(show_plots, r_c1, r_c2, valid1, valid2, faultMode):
    """ Test the faultdetection module. Setup a simulation """

    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))  # Add a new task to the process

    # Construct the ephemNavConverter module
    # Set the names for the input messages
    faults = faultDetection.faultDetection()
    faults.sigmaFault = 3
    faults.faultMode = faultMode
    # ephemNavConfig.outputState = simFswInterfaceMessages.NavTransIntMsg()

    # This calls the algContain to setup the selfInit, update, and reset
    faults.ModelTag = "faultDet"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, faults)

    # Create the input messages.
    inputPrimary = messaging.OpNavMsgPayload()
    inputSecondary = messaging.OpNavMsgPayload()
    inputCamera = messaging.CameraConfigMsgPayload()
    inputAtt = messaging.NavAttMsgPayload()

    # Set camera
    inputCamera.fieldOfView = 2.0 * np.arctan(10*1e-3 / 2.0 / (1.*1e-3) )  # 2*arctan(s/2 / f)
    inputCamera.resolution = [512, 512]
    inputCamera.sigma_CB = [1.,0.3,0.1]
    camInMsg = messaging.CameraConfigMsg().write(inputCamera)
    faults.cameraConfigInMsg.subscribeTo(camInMsg)

    # Set attitude
    inputAtt.sigma_BN = [0.6, 1., 0.1]
    attInMsg = messaging.NavAttMsg().write(inputAtt)
    faults.attInMsg.subscribeTo(attInMsg)

    BN = rbk.MRP2C(inputAtt.sigma_BN)
    CB = rbk.MRP2C(inputCamera.sigma_CB)
    NC = np.dot(BN.T, CB.T)
    # Set primary
    inputPrimary.r_BN_C = r_c1
    inputPrimary.r_BN_N = np.dot(NC, np.array(r_c1)).tolist()
    inputPrimary.valid = valid1
    inputPrimary.covar_C = [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
    inputPrimary.covar_N = np.dot(np.dot(NC, np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]).reshape([3,3])), NC.T).flatten().tolist()
    inputPrimary.timeTag = 12345
    op1InMsg = messaging.OpNavMsg().write(inputPrimary)
    faults.navMeasPrimaryInMsg.subscribeTo(op1InMsg)

    # Set secondary
    inputSecondary.r_BN_C = r_c2
    inputSecondary.r_BN_N = np.dot(NC, np.array(r_c2)).tolist()
    inputSecondary.valid = valid2
    inputSecondary.covar_C = [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
    inputSecondary.covar_N = np.dot(np.dot(NC, np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]).reshape([3,3])), NC.T).flatten().tolist()
    inputSecondary.timeTag = 12345
    op2InMsg = messaging.OpNavMsg().write(inputSecondary)
    faults.navMeasSecondaryInMsg.subscribeTo(op2InMsg)

    dataLog = faults.opNavOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()
    # The result isn't going to change with more time. The module will continue to produce the same result
    unitTestSim.ConfigureStopTime(testProcessRate)  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # Truth Vlaues
    faultDetectedTrue= 0
    timTagExp = inputPrimary.timeTag
    if valid1 ==0 and valid2==0:
        timTagExp = 0
        r_Cexp = [0,0,0]
        covar_Cexp = [0,0,0,0,0,0,0,0,0]
    if valid1 == 0  and valid2 ==1:
        if faultMode > 0:
            timTagExp = 0
            r_Cexp = [0, 0, 0]
            covar_Cexp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if faultMode == 0:
            r_Cexp = r_c2
            covar_Cexp =  [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
    if valid1 == 1  and valid2 ==0:
        if faultMode < 2:
            r_Cexp = r_c1
            covar_Cexp =  [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
        if faultMode == 2:
            timTagExp = 0
            r_Cexp = [0, 0, 0]
            covar_Cexp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if valid1 == 1  and valid2 ==1:
        r1 = np.array(r_c1)
        r2 = np.array(r_c2)
        faultNorm = np.linalg.norm(r2 - r1)

        covarz = np.array([0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]).reshape([3,3])
        z1 = covarz[:,2]
        z2 = np.copy(z1)
        if (faultNorm > faults.sigmaFault*(np.linalg.norm(z1)+np.linalg.norm(z2))):
            faultDetectedTrue = 1
            r_Cexp = r_c2
            covar_Cexp = [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
        elif faultMode > 0:
            r_Cexp = r_c1
            covar_Cexp =  [0.5, 0., 0., 0., 0.5, 0., 0., 0., 1.]
        elif faultMode == 0:
            covar_Cexp = np.linalg.inv(np.linalg.inv(covarz) + np.linalg.inv(covarz)).flatten().tolist()
            r_Cexp = np.dot(np.linalg.inv(np.linalg.inv(covarz) + np.linalg.inv(covarz)), np.dot(np.linalg.inv(covarz), r1) +  np.dot(np.linalg.inv(covarz), r2)).tolist()


    posErr = 1e-10
    print(posErr)

    outputR = dataLog.r_BN_C
    outputCovar = dataLog.covar_C
    outputTime = dataLog.timeTag
    detected = dataLog.faultDetected

    #
    #
    for i in range(len(outputR[-1, 1:])):
        if np.abs(r_Cexp[i] - outputR[-1, i]) > 1E-10 or np.isnan(outputR.any()):
            testFailCount += 1
            testMessages.append("FAILED: Position Check in pixelLine")

    for i in range(len(outputCovar[-1, 0:])):
        if np.abs((covar_Cexp[i] - outputCovar[-1, i])) > 1E-10 or np.isnan(outputTime.any()):
            testFailCount += 1
            testMessages.append("FAILED: Covar Check in pixelLine")

    if np.abs((timTagExp - outputTime[-1])) > 1E-10 or np.isnan(outputTime.any()):
        testFailCount += 1
        testMessages.append("FAILED: Time Check in pixelLine")

    if np.abs(faultDetectedTrue - detected[-1]) > 1E-10 or np.isnan(outputTime.any()):
        testFailCount += 1
        testMessages.append("FAILED: Time Check in pixelLine")
    #
    #   print out success message if no error were found
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + faults.ModelTag)
    else:
        colorText = 'Red'
        print("Failed: " + faults.ModelTag)


    return [testFailCount, ''.join(testMessages)]


if __name__ == '__main__':
    faultdetection(False ,[10, 10, 1], [11, 9, 0.5], 1, 1, 1)
