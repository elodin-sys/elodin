#
#   Unit Test Script
#   Module Name:        pixelLineConverter.py
#   Creation Date:      May 16, 2019
#   Author:             Thibaud Teil
#

import inspect
import os

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import horizonOpNav
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import SimulationBaseClass, unitTestSupport, macros

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

def back_substitution(A, b):
    n = b.size
    x = np.zeros_like(b)

    if A[-1, -1] == 0:
        raise ValueError

    x[-1] = b[-1]/ A[-1, -1]
    for i in range(n-2, -1, -1):
        sum=0
        for j in range(i, n):
            sum += A[i, j]*x[j]
        x[i] = (b[i] - sum)/A[i,i]
    return x


def test_horizonOpNav():
    """
    Unit test for Horizon Navigation. The unit test specifically covers:

        1. Individual methods: This module contains a back substitution method as well as a QR decomposition.
            This test ensures that they are working properly with a direct test of the method input/outputs with
            expected results

        2. State and Covariances: This unit test also computes the state estimate and covariance in python. This is
            compared directly to the output from the module for exact matching.

    The Horizon Nav module gives the spacecraft position given a limb input. This test ensures that the results are as
    expected both for the state estimate and the covariance associated with the measurement.
    """
    [testResults, testMessage] = horizonOpNav_methods()
    assert testResults < 1, testMessage
    [testResults, testMessage] = horizonOpNav_update()
    assert testResults < 1, testMessage

def horizonOpNav_methods():
    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    ###################################################################################
    ## Testing QR decomp
    ###################################################################################
    Hinput = np.array([[1,2,3],[1,20,3],[3,0,1],[2,1,0],[20,-1, -5],[0,10,-5]])
    numStates = np.shape(Hinput)[0]
    # Fill in the variables for the test
    Qin = horizonOpNav.new_doubleArray(3 * numStates)
    Rin = horizonOpNav.new_doubleArray(3 * 3)
    Hin = horizonOpNav.new_doubleArray(numStates * 3)
    for j in range(numStates*3):
        horizonOpNav.doubleArray_setitem(Qin, j, 0)
    for j in range(3 * 3):
        horizonOpNav.doubleArray_setitem(Rin, j, 0)
    for j in range(numStates * 3):
        horizonOpNav.doubleArray_setitem(Hin, j, Hinput.flatten().tolist()[j])
    horizonOpNav.QRDecomp(Hin, numStates, Qin, Rin)

    Qout = []
    for j in range(3 * numStates):
        Qout.append(horizonOpNav.doubleArray_getitem(Qin, j))
    Rout = []
    for j in range(3 * 3):
        Rout.append(horizonOpNav.doubleArray_getitem(Rin, j))

    q,r = np.linalg.qr(Hinput)

    Rpy = np.zeros([3,3])
    Qpy = np.zeros([numStates, 3])
    for i in range(0,3):
        Qpy[:,i] = Hinput[:,i]
        for j in range(i):
            Rpy[j,i] = np.dot(Qpy[:,j], Hinput[:,i])
            Qpy[:,i]= Qpy[:,i] - Rpy[j,i]*Qpy[:,j]
        Rpy[i,i] = np.linalg.norm(Qpy[:,i])
        Qpy[:,i] = 1 / Rpy[i,i] *  Qpy[:,i]


    Qtest = np.array(Qout).reshape([numStates,3])
    Rtest = np.array(Rout).reshape(3, 3)
    errorNorm1 = np.linalg.norm(Qpy - Qtest)
    errorNorm2 = np.linalg.norm(Rpy - Rtest)
    if (errorNorm1 > 1.0E-10):
        print(errorNorm1, "QR decomp")
        testFailCount += 1
        testMessages.append("QR decomp Failure in Q" + "\n")
    if (errorNorm2 > 1.0E-10):
        print(errorNorm2, "QR decomp")
        testFailCount += 1
        testMessages.append("QR decomp Failure in R" + "\n")
    errorNorm1 = np.linalg.norm(q + Qtest)
    errorNorm2 = np.linalg.norm(r[:3,:3] + Rtest)
    if (errorNorm1 > 1.0E-10):
        print(errorNorm1, "QR decomp")
        testFailCount += 1
        testMessages.append("QR decomp Failure in Q" + "\n")
    if (errorNorm2 > 1.0E-10):
        print(errorNorm2, "QR decomp")
        testFailCount += 1
        testMessages.append("QR decomp Failure in R" + "\n")

    ###################################################################################
    ## Testing Back Sub
    ###################################################################################
    V = np.ones(3)
    nIn = horizonOpNav.new_doubleArray(3)
    VIn = horizonOpNav.new_doubleArray(3)
    RIn = horizonOpNav.new_doubleArray(numStates*3)
    for i in range(3):
        horizonOpNav.doubleArray_setitem(nIn, i, 0.0)
    for i in range(3*3):
        horizonOpNav.doubleArray_setitem(RIn, i, r.flatten().tolist()[i])
    for i in range(3):
        horizonOpNav.doubleArray_setitem(VIn, i, V.flatten().tolist()[i])

    horizonOpNav.BackSub(RIn, VIn, 3, nIn)
    BackSubOut = []
    for i in range(3):
        BackSubOut.append(horizonOpNav.doubleArray_getitem(nIn, i))

    exp = back_substitution(r[:3,:3], V)

    BackSubOut = np.array(BackSubOut)
    errorNorm = np.linalg.norm(exp - BackSubOut)
    if(errorNorm > 1.0E-10):
        print(errorNorm, "BackSub")
        testFailCount += 1
        testMessages.append("BackSub Failure " + "\n")

    return [testFailCount, ''.join(testMessages)]

###################################################################################
## Testing dynamics matrix computation
###################################################################################
def horizonOpNav_update():
    # Create a sim module as an empty container
    testFailCount = 0  # zero unit test result counter
    testMessages = []  # create empty array to store test log messages
    unitTaskName = "unitTask"  # arbitrary name (don't change)
    unitProcessName = "TestProcess"  # arbitrary name (don't change)
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    testProcessRate = macros.sec2nano(0.5)  # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))  # Add a new task to the process

    # Construct the ephemNavConverter module
    # Set the names for the input messages
    opNav = horizonOpNav.horizonOpNav()
    opNav.noiseSF = 2
    # ephemNavConfig.outputState = simFswInterfaceMessages.NavTransIntMsg()

    # This calls the algContain to setup the selfInit, update, and reset
    opNav.ModelTag = "limbNav"

    # Add the module to the task
    unitTestSim.AddModelToTask(unitTaskName, opNav)

    # These are example points for fitting used from an image processing algorithm
    inputPoints = [226., 113., 227., 113., 223., 114., 224., 114., 225., 114., 219.,
       115., 220., 115., 221., 115., 222., 115., 215., 116., 216., 116.,
       217., 116., 218., 116., 212., 117., 213., 117., 214., 117., 209.,
       118., 210., 118., 211., 118., 205., 119., 206., 119., 207., 119.,
       208., 119., 204., 120., 205., 120., 201., 121., 202., 121., 203.,
       121., 199., 122., 200., 122., 197., 123., 198., 123., 195., 124.,
       196., 124., 193., 125., 194., 125., 191., 126., 192., 126., 189.,
       127., 190., 127., 187., 128., 188., 128., 185., 129., 186., 129.,
       183., 130., 184., 130., 181., 131., 182., 131., 180., 132., 181.,
       132., 178., 133., 179., 133., 177., 134., 178., 134., 175., 135.,
       176., 135., 174., 136., 175., 136., 172., 137., 173., 137., 171.,
       138., 172., 138., 170., 139., 171., 139., 168., 140., 169., 140.,
       167., 141., 168., 141., 166., 142., 167., 142., 164., 143., 165.,
       143., 163., 144., 164., 144., 162., 145., 163., 145., 161., 146.,
       162., 146., 160., 147., 161., 147., 159., 148., 160., 148., 158.,
       149., 159., 149., 156., 150., 157., 150., 155., 151., 156., 151.,
       154., 152., 155., 152., 153., 153., 154., 153., 153., 154., 152.,
       155., 151., 156., 152., 156., 150., 157., 151., 157., 149., 158.,
       150., 158., 148., 159., 149., 159., 147., 160., 148., 160., 146.,
       161., 147., 161., 145., 162., 146., 162., 145., 163., 144., 164.,
       143., 165., 144., 165., 142., 166., 143., 166., 142., 167., 141.,
       168., 140., 169., 141., 169., 139., 170., 140., 170., 139., 171.,
       138., 172., 137., 173., 138., 173., 137., 174., 136., 175., 135.,
       176., 136., 176., 135., 177., 134., 178., 133., 179., 134., 179.,
       133., 180., 132., 181., 132., 182., 131., 183., 131., 184., 130.,
       185., 129., 186., 130., 186., 129., 187., 128., 188., 128., 189.,
       127., 190., 127., 191., 126., 192., 126., 193., 125., 194., 125.,
       195., 125., 196., 124., 197., 124., 198., 123., 199., 123., 200.,
       122., 201., 122., 202., 122., 203., 121., 204., 120., 205., 121.,
       205., 120., 206., 120., 207., 120., 208., 119., 209., 119., 210.,
       119., 211., 118., 212., 118., 213., 118., 214., 117., 215., 117.,
       216., 117., 217., 117., 218., 116., 219., 116., 220., 116., 221.,
       116., 222., 115., 223., 115., 224., 115., 225., 115., 226., 114.,
       227., 114., 228., 114., 229., 114., 230., 114., 231., 114., 232.,
       113., 233., 113., 234., 113., 235., 113., 236., 113., 237., 113.,
       238., 113., 239., 112., 240., 112., 241., 112., 242., 112., 243.,
       112., 244., 112., 245., 112., 246., 112., 247., 112., 248., 112.,
       249., 112., 250., 112., 251., 112., 252., 112., 253., 112., 254.,
       111., 255., 111., 256., 112., 257., 112., 258., 112., 259., 112.,
       260., 112., 261., 112., 262., 112., 263., 112., 264., 112., 265.,
       112., 266., 112., 267., 112., 268., 112., 269., 112., 270., 112.,
       271., 113., 272., 113., 273., 113., 274., 113., 275., 113., 276.,
       113., 277., 113., 278., 114., 279., 114., 280., 114., 281., 114.,
       282., 114., 283., 114., 284., 115., 285., 115., 286., 115., 287.,
       115., 288., 116., 289., 116., 290., 116., 291., 116., 292., 117.,
       293., 117., 294., 117., 295., 117., 296., 118., 297., 118., 298.,
       118., 299., 119., 300., 119., 301., 119., 302., 120., 303., 120.,
       304., 120., 305., 121., 306., 121., 307., 122., 308., 122., 309.,
       122., 310., 123., 311., 123., 312., 124., 313., 124., 314., 125.,
       315., 125., 316., 125., 317., 126., 318., 126., 319., 127., 320.,
       127., 321., 128., 322., 128., 323., 129., 324., 129., 325., 130.,
       325., 130., 326., 131., 327., 131., 328., 132., 329., 132., 330.,
       133., 331., 133., 332., 134., 332., 134., 333., 135., 334., 135.,
       335., 136., 335., 136., 336., 137., 337., 137., 338., 138., 338.,
       138., 339., 139., 340., 139., 341., 140., 341., 140., 342., 141.,
       342., 141., 343., 142., 344., 142., 345., 143., 345., 143., 346.,
       144., 346., 144., 347., 145., 348., 145., 349., 146., 349., 146.,
       350., 147., 350., 147., 351., 148., 351., 148., 352., 149., 352.,
       149., 353., 150., 353., 150., 354., 151., 354., 151., 355., 152.,
       356., 152., 357., 153., 357., 153., 358., 154., 358., 154., 359.,
       155., 359., 155., 360., 156., 360., 156., 361., 157., 361., 158.,
       362., 159., 362., 159., 363., 160., 363., 160., 364., 161., 364.,
       161., 365., 162., 365., 162., 366., 163., 366., 163., 367., 164.,
       367., 164., 368., 165., 368., 166., 369., 167., 369., 167., 370.,
       168., 370., 168., 371., 169., 371., 169., 372., 170., 372., 171.,
       373., 172., 373., 172., 374., 173., 374., 174., 375., 175., 375.,
       175., 376., 176., 376., 177., 377., 178., 377., 178., 378., 179.,
       378., 180., 379., 181., 379., 181., 380., 182., 380., 183., 381.,
       184., 381., 185., 382., 186., 382., 187., 383., 188., 383., 188.,
       384., 189., 384., 190., 385., 191., 385., 192., 386.]

    # Create the input messages.
    inputCamera = messaging.CameraConfigMsgPayload()
    inputLimbMsg = messaging.OpNavLimbMsgPayload()
    inputAtt = messaging.NavAttMsgPayload()

    # Set camera
    inputCamera.fieldOfView = 2.0 * np.arctan(10*1e-3 / 2.0 / 1. )  # 2*arctan(s/2 / f)
    inputCamera.resolution = [512, 512]
    inputCamera.sigma_CB = [1.,0.2,0.3]
    camInMsg = messaging.CameraConfigMsg().write(inputCamera)
    opNav.cameraConfigInMsg.subscribeTo(camInMsg)

    # Set circles
    inputLimbMsg.valid = 1
    inputLimbMsg.limbPoints = inputPoints
    inputLimbMsg.numLimbPoints = int(len(inputPoints)/2)
    inputLimbMsg.timeTag = 12345
    limbInMsg = messaging.OpNavLimbMsg().write(inputLimbMsg)
    opNav.limbInMsg.subscribeTo(limbInMsg)


    # Set attitude
    inputAtt.sigma_BN = [0.6, 1., 0.1]
    attInMsg = messaging.NavAttMsg().write(inputAtt)
    opNav.attInMsg.subscribeTo(attInMsg)


    # Set module for Mars
    opNav.planetTarget = 2
    dataLog = opNav.opNavOutMsg.recorder()
    unitTestSim.AddModelToTask(unitTaskName, dataLog)

    # Initialize the simulation
    unitTestSim.InitializeSimulation()
    # The result isn't going to change with more time. The module will continue to produce the same result
    unitTestSim.ConfigureStopTime(testProcessRate)  # seconds to stop simulation
    unitTestSim.ExecuteSimulation()

    # Truth Vlaues
    ############################
    Q = np.eye(3)
    B = np.zeros([3,3])
    Q *= 1/(3396.19*1E3)  # km
    # Q[2,2] = 1/(3376.2*1E3)

    numPoints = int(len(inputPoints)/2)

    CB = rbk.MRP2C(inputCamera.sigma_CB)
    BN = rbk.MRP2C(inputAtt.sigma_BN)
    CN = np.dot(CB,BN)
    B = np.dot(Q, CN.T)

    # Transf camera to meters
    alpha =0
    up = inputCamera.resolution[0] / 2
    vp = inputCamera.resolution[1] / 2
    pX = 2. * np.tan(inputCamera.fieldOfView * inputCamera.resolution[0] / inputCamera.resolution[1] / 2.0)
    pY = 2. * np.tan(inputCamera.fieldOfView / 2.0)
    d_x = inputCamera.resolution[0] / pX
    d_y = inputCamera.resolution[1] / pY

    transf = np.zeros([3,3])
    transf[0, 0] = 1 / d_x
    transf[1, 1] = 1 / d_y
    transf[2, 2] = 1
    transf[0, 1] = -alpha/(d_x*d_y)
    transf[0, 2] = (alpha*vp - d_y*up)/ (d_x * d_y)
    transf[1, 2] = -vp / (d_y)

    s = np.zeros([numPoints,3])
    sBar = np.zeros([numPoints,3])
    sBarPrime = np.zeros([numPoints,3])
    H = np.zeros([numPoints,3])
    for i in range(numPoints):
        s[i,:] = np.dot(transf, np.array([inputPoints[2*i], inputPoints[2*i+1], 1]))
        sBar[i,:] = np.dot(B, s[i,:])
        sBarPrime[i,:] = sBar[i,:]/np.linalg.norm(sBar[i,:])
        H[i,:] = sBarPrime[i,:]

    # QR H
    Rpy = np.zeros([3,3])
    Qpy = np.zeros([numPoints, 3])
    for i in range(0,3):
        Qpy[:,i] = H[:,i]
        for j in range(i):
            Rpy[j,i] = np.dot(Qpy[:,j], H[:,i])
            Qpy[:,i]= Qpy[:,i] - Rpy[j,i]*Qpy[:,j]
        Rpy[i,i] = np.linalg.norm(Qpy[:,i])
        Qpy[:,i] = 1 / Rpy[i,i] *  Qpy[:,i]

    errorNorm1 = np.linalg.norm(np.dot(Qpy, Rpy) - H)
    if (errorNorm1 > 1.0E-8):
        print(errorNorm1, "QR decomp")
        testFailCount += 1
        testMessages.append("QR decomp Failure in update test " + "\n")

    # Back Sub
    RHS = np.dot(Qpy.T, np.ones(numPoints))
    n = back_substitution(Rpy, RHS)
    n_test = np.dot(np.linalg.inv(Rpy), RHS)

    R_s = (opNav.noiseSF*inputCamera.resolution[0]/(numPoints))**2/d_x**2*np.array([[1,0,0],[0,1,0],[0,0,0]])
    R_s = np.dot(np.dot(B, R_s), B.T)
    R_yInv = np.zeros([numPoints, numPoints])
    for i in range(numPoints):
        J = 1./np.linalg.norm(sBar[i,:])*np.dot(n, np.eye(3) - np.outer(sBarPrime[i,:], sBarPrime[i,:]))
        temp = np.dot(R_s, J)
        R_yInv[i,i] = 1./np.dot(temp, J)

    Pn = np.linalg.inv(np.dot(np.dot(H.T, R_yInv),H))
    F = -(np.dot(n,n) - 1)**(-0.5)*np.dot(np.linalg.inv(B), np.eye(3) - np.outer(n,n)/(np.dot(n,n)-1))
    Covar_C_test = np.dot(np.dot(F, Pn), F.T)
    errorNorm1 = np.linalg.norm(n_test - n)
    if (errorNorm1 > 1.0E-8):
        print(errorNorm1, "Back Sub")
        testFailCount += 1
        testMessages.append("Back Sub Failure in update test " + "\n")

    r_BN_C = - (np.dot(n,n) - 1.)**(-0.5)*np.dot(np.linalg.inv(B), n)

    posErr = 1e-3 #(m)
    covarErr = 1e-5
    unitTestSupport.writeTeXSnippet("toleranceValuePos", str(posErr), path)
    unitTestSupport.writeTeXSnippet("toleranceValueVel", str(covarErr), path)

    outputR = dataLog.r_BN_C
    outputCovar = dataLog.covar_C
    outputTime = dataLog.timeTag

    for i in range(len(outputR[-1, 1:])):
        if np.abs((r_BN_C[i] - outputR[0, i])/r_BN_C[i]) > posErr or np.isnan(outputR.any()):
            testFailCount += 1
            testMessages.append("FAILED: Position Check in Horizon Nav for index "+ str(i) + " with error " + str(np.abs((r_BN_C[i] - outputR[-1, i+1])/r_BN_C[i])))

    for i in range(len(outputCovar[-1, 1:])):
        if np.abs((Covar_C_test.flatten()[i] - outputCovar[0, i])/Covar_C_test.flatten()[i]) > covarErr or np.isnan(outputTime.any()):
            testFailCount += 1
            testMessages.append("FAILED: Covar Check in Horizon Nav for index "+ str(i) + " with error " + str(np.abs((Covar_C_test.flatten()[i] - outputCovar[-1, i+1]))))

    snippentName = "passFail"
    if testFailCount == 0:
        colorText = 'ForestGreen'
        print("PASSED: " + opNav.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "PASSED" + '}'
    else:
        colorText = 'Red'
        print("Failed: " + opNav.ModelTag)
        passedText = r'\textcolor{' + colorText + '}{' + "Failed" + '}'
        print(testMessages)
    unitTestSupport.writeTeXSnippet(snippentName, passedText, path)


    return [testFailCount, ''.join(testMessages)]


if __name__ == '__main__':
    # horizonOpNav_methods()
    horizonOpNav_update()
