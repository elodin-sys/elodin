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
#   Module Name:        constrainedAttitudeManeuver
#   Author:             Riccardo Calaon
#   Creation Date:      April 16, 2022
#


import os

import numpy as np
import pytest
from Basilisk.architecture import BSpline
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import constrainedAttitudeManeuver
from Basilisk.utilities import RigidBodyKinematics as rbk
# Import all of the modules that we are going to be called in this simulation
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import unitTestSupport

path = os.path.dirname(os.path.abspath(__file__))
dataFileName = None


def shadowSetMap(sigma, switch):
    sigma = np.array(sigma)
    if switch:
        s2 = (sigma[0]**2 + sigma[1]**2 + sigma[2]**2)
        if not s2 == 0:
            return -sigma / s2
        else:
            return sigma
    else:
        return sigma


class constraint:
    def __init__(self, axis, color):
        self.axis =  axis / np.linalg.norm(axis)
        self.color = color
        

class node:
    def __init__(self, sigma_BN, constraints, **kwargs):
        self.sigma_BN = np.array(sigma_BN)
        s = np.linalg.norm(self.sigma_BN)
        if s > 1:
            self.sigma_BN = shadowSetMap(sigma_BN, True)  # mapping to shadow set if |sigma| > 1
        if np.abs(s-1) < 1e-5:
            s = 1.
        self.isBoundary = False
        self.s = s
        if s == 1:
            self.isBoundary = True
        self.isFree = True
        self.color = ''
        self.neighbors = {}
        self.heuristic = 0
        self.priority = 0
        self.backpointer = self
        # check cosntraint compliance
        sigma_tilde = np.array([ [0,           -sigma_BN[2], sigma_BN[1]],
                                 [sigma_BN[2],      0,      -sigma_BN[0]],
                                 [-sigma_BN[1], sigma_BN[0],     0      ] ])
        BN = np.identity(3) + ( 8*np.matmul(sigma_tilde, sigma_tilde) -4*(1-s**2)*sigma_tilde ) / (1 + s**2)**2
        NB = BN.transpose()
        # checking for keepOut constraint violation
        if 'keepOut_b' in kwargs:
            for i in range(len(kwargs['keepOut_b'])):
                b_B = np.array(kwargs['keepOut_b'][i])
                b_N = np.matmul(NB, b_B)
                for c in constraints['keepOut']:
                    if np.dot(b_N, c.axis) >= np.cos(kwargs['keepOut_fov'][i]):
                        self.isFree = False
                        self.color = c.color
                        return
        # checking for keepIn constraint violation (at least one SS must see the Sun)
        if 'keepIn_b' in kwargs:
            b_N = []
            for i in range(len(kwargs['keepIn_b'])):
                b_B = np.array(kwargs['keepIn_b'][i])
                b_N.append(np.matmul(NB, b_B))
            isIn = False
            for c in constraints['keepIn']:
                for i in range(len(b_N)):
                    if np.dot(b_N[i], c.axis) >= np.cos(kwargs['keepIn_fov'][i]):
                        isIn = True
                        self.color = c.color
            if not isIn:
                self.isFree = False


def distanceCart(n1, n2):
    d1 = np.linalg.norm(n1.sigma_BN - n2.sigma_BN)
    sigma1norm2 = n1.sigma_BN[0]**2 + n1.sigma_BN[1]**2 + n1.sigma_BN[2]**2
    sigma2norm2 = n2.sigma_BN[0]**2 + n2.sigma_BN[1]**2 + n2.sigma_BN[2]**2
    if sigma2norm2 > 1e-8:
        sigma2_SS = shadowSetMap(n2.sigma_BN, True)
        d2 = np.linalg.norm(n1.sigma_BN - sigma2_SS)
    else:
        d2 = d1
    if sigma1norm2 > 1e-8:
        sigma1_SS = shadowSetMap(n1.sigma_BN, True)
        d3 = np.linalg.norm(sigma1_SS - n2.sigma_BN)
    else:
        d3 = d1
    if sigma1norm2 > 1e-8 and sigma2norm2 > 1e-8:
        d4 = np.linalg.norm(sigma1_SS - sigma2_SS)
    else:
        d4 = d1

    return min(d1, d2, d3, d4)


def distanceMRP(n1, n2):    
    s1 = n1.sigma_BN
    s2 = n2.sigma_BN
    sigma1norm2 = n1.sigma_BN[0]**2 + n1.sigma_BN[1]**2 + n1.sigma_BN[2]**2
    sigma2norm2 = n2.sigma_BN[0]**2 + n2.sigma_BN[1]**2 + n2.sigma_BN[2]**2

    D = 1 + (sigma1norm2*sigma2norm2)**2 + 2*np.dot(s1, s2)

    if abs(D) < 1e-5:
        s2 = shadowSetMap(s2, True)
        sigma2norm2 = 1 / sigma2norm2
        D = 1 + (sigma1norm2*sigma2norm2)**2 + 2*np.dot(s1, s2)

    s12 = ( (1-sigma2norm2)*s1 - (1-sigma1norm2)*s2 + 2*np.cross(s1, s2) ) / D
    sigma12norm2 = np.linalg.norm(s12)**2

    if sigma12norm2 > 1:
        s12 = shadowSetMap(s12, True)

    return 4*np.arctan(np.linalg.norm(s12))


def mirrorFunction(i, j, k):
    return [ [i, j, k], [-i, j, k], [i, -j, k], [i, j, -k], [-i, -j, k], [-i, j, -k], [i, -j, -k], [-i, -j, -k] ]


def neighboringNodes(i, j, k):
    return [ [i-1,  j,  k], [i+1,  j,  k], [i,  j-1,  k], [i,  j+1,  k], [i,  j,  k-1], [i,  j,  k+1],
             [i-1, j-1, k], [i+1, j-1, k], [i-1, j+1, k], [i+1, j+1, k], [i-1, j, k-1], [i+1, j, k-1],
             [i-1, j, k+1], [i+1, j, k+1], [i, j-1, k-1], [i, j+1, k-1], [i, j-1, k+1], [i, j+1, k+1],
             [i-1,j-1,k-1], [i+1,j-1,k-1], [i-1,j+1,k-1], [i-1,j-1,k+1], [i+1,j+1,k-1], [i+1,j-1,k+1], [i-1,j+1,k+1], [i+1,j+1,k+1] ]


def generateGrid(n_start, n_goal, N, constraints, data):
    
    u = np.linspace(0, 1, N, endpoint = True)
    nodes = {}

    # add internal nodes (|sigma| <= 1)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if (u[i]**2+u[j]**2+u[k]**2) <= 1:
                    for m in mirrorFunction(i, j, k):
                        if (m[0], m[1], m[2]) not in nodes:
                            nodes[(m[0], m[1], m[2])] = node([np.sign(m[0])*u[i], np.sign(m[1])*u[j], np.sign(m[2])*u[k]], constraints, **data)
    # add missing boundary nodes (|sigma| = 1)
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
                if (i, j, k) in nodes:
                    if not nodes[(i, j, k)].isBoundary:
                        if (i+1, j, k) not in nodes:
                            for m in mirrorFunction(i+1, j, k):
                                if (m[0], m[1], m[2]) not in nodes:
                                    nodes[(m[0], m[1], m[2])] = node([np.sign(m[0])*(1-u[j]**2-u[k]**2)**0.5, np.sign(m[1])*u[j], np.sign(m[2])*u[k]], constraints, **data)
                        if (i, j+1, k) not in nodes:
                            for m in mirrorFunction(i, j+1, k):
                                if (m[0], m[1], m[2]) not in nodes:
                                    nodes[(m[0], m[1], m[2])] = node([np.sign(m[0])*u[i], np.sign(m[1])*(1-u[i]**2-u[k]**2)**0.5, np.sign(m[2])*u[k]], constraints, **data)
                        if (i, j, k+1) not in nodes:
                            for m in mirrorFunction(i, j, k+1):
                                if (m[0], m[1], m[2]) not in nodes:
                                    nodes[(m[0], m[1], m[2])] = node([np.sign(m[0])*u[i], np.sign(m[1])*u[j], np.sign(m[2])*(1-u[i]**2-u[j]**2)**0.5], constraints, **data)
    
    # link nodes
    for key1 in nodes:
        i = key1[0]
        j = key1[1]
        k = key1[2]
        # link nodes to immediate neighbors
        for n in neighboringNodes(i, j, k):
            key2 = (n[0], n[1], n[2])
            if key2 in nodes:
                if nodes[key1].isFree and nodes[key2].isFree:
                    nodes[key1].neighbors[key2] = nodes[key2]
        # link boundary nodes to neighbors of respective shadow sets
        if nodes[key1].isBoundary:
            if (-i, -j, -k) in nodes:
                for key2 in nodes[(-i, -j, -k)].neighbors:
                    if nodes[key1].isFree and nodes[key2].isFree and key2 not in nodes[key1].neighbors:
                        nodes[key1].neighbors[key2] = nodes[key2]

    # add start and goal nodes
    # looking for closest nodes to start and goal
    ds = 10
    dg = 10
    for key in nodes:
        if nodes[key].isFree:
            d1 = distanceCart(nodes[key], n_start)
            if abs(d1-ds) < 1e-6:
                if np.linalg.norm(nodes[key].sigma_BN) < np.linalg.norm(nodes[key_s].sigma_BN):
                    ds = d1
                    n_s = nodes[key]
                    key_s = key
            else:
                if d1 < ds:
                    ds = d1
                    n_s = nodes[key]
                    key_s = key
            d2 = distanceCart(nodes[key], n_goal)
            if abs(d2-dg) < 1e-6:
                if np.linalg.norm(nodes[key].sigma_BN) < np.linalg.norm(nodes[key_g].sigma_BN):
                    dg = d2
                    n_g = nodes[key]
                    key_g = key
            else:
                if d2 < dg:
                    dg = d2
                    n_g = nodes[key]
                    key_g = key
    for key in n_s.neighbors:
        n_start.neighbors[key] = n_s.neighbors[key]
    for key in n_g.neighbors:
        nodes[key].neighbors[key_g] = n_goal
    nodes[key_s] = n_start 
    nodes[key_g] = n_goal

    return nodes


def backtrack(n, n_start):
    if n == n_start:
        path = [n]
        return path
    else:
        path = backtrack(n.backpointer, n_start)
        path.append(n)
        return path


def pathHandle(path, avgOmega):

    T = [0]
    S = 0
    for n in range(len(path)-1):
        T.append(T[n] + distanceCart(path[n], path[n+1]))
        S += T[n+1] - T[n]

    X1 = []
    X2 = []
    X3 = []

    shadowSet = False
    for n in range(len(path)-1):
        if not shadowSet:
            sigma = path[n].sigma_BN
        else:
            if unitTestSupport.isVectorEqual(path[n].sigma_BN, [0,0,0], 1e-6):
                for m in range(0,n):
                    s2 = (X1[m]**2 + X2[m]**2 + X3[m]**2)**0.5
                    X1[m] = -X1[m] / s2
                    X2[m] = -X2[m] / s2
                    X3[m] = -X3[m] / s2
                shadowSet = not shadowSet
            sigma = rbk.MRPswitch(path[n].sigma_BN, 0)
        delSigma = path[n+1].sigma_BN - path[n].sigma_BN
        if (np.linalg.norm(delSigma) > 1):
            shadowSet = not shadowSet
        X1.append(sigma[0])
        X2.append(sigma[1])
        X3.append(sigma[2])
    if shadowSet:
        sigma = rbk.MRPswitch(path[-1].sigma_BN, 0)
    else:
        sigma = path[-1].sigma_BN
    X1.append(sigma[0])
    X2.append(sigma[1])
    X3.append(sigma[2])

    Input = BSpline.InputDataSet(X1, X2, X3)
    Input.setT( np.array(T) * 4 * S / (T[-1] * avgOmega) )

    return Input


def spline(Input, omegaS, omegaG):

    sigmaS = [Input.X1[0][0],  Input.X2[0][0],  Input.X3[0][0]]
    sigmaG = [Input.X1[-1][0], Input.X2[-1][0], Input.X3[-1][0]]
    sigmaDotS = rbk.dMRP(sigmaS, omegaS)
    sigmaDotG = rbk.dMRP(sigmaG, omegaG)

    Input.setXDot_0(sigmaDotS)
    Input.setXDot_N(sigmaDotG)

    Output = BSpline.OutputDataSet()
    BSpline.interpolate(Input, 100, 4, Output)

    return Output


def computeTorque(sigma, sigmaDot, sigmaDDot, I):

    omega = rbk.dMRP2Omega(sigma, sigmaDot)
    omegaDot = rbk.ddMRP2dOmega(sigma, sigmaDot, sigmaDDot)

    return np.matmul(I, omegaDot) + np.cross(omega, np.matmul(I, omega))


def effortEvaluation(Output, I):

    effort = 0
    sigma     = [Output.X1[0][0], Output.X2[0][0], Output.X3[0][0]]
    sigmaDot  = [Output.XD1[0][0], Output.XD2[0][0], Output.XD3[0][0]]
    sigmaDDot = [Output.XDD1[0][0], Output.XDD2[0][0], Output.XDD3[0][0]]
    L_a = computeTorque(sigma, sigmaDot, sigmaDDot, I)

    for n in range(len(Output.T)-1):
        sigma     = [Output.X1[n+1][0], Output.X2[n+1][0], Output.X3[n+1][0]]
        sigmaDot  = [Output.XD1[n+1][0], Output.XD2[n+1][0], Output.XD3[n+1][0]]
        sigmaDDot = [Output.XDD1[n+1][0], Output.XDD2[n+1][0], Output.XDD3[n+1][0]]
        L_b = computeTorque(sigma, sigmaDot, sigmaDDot, I)
        effort += (np.linalg.norm(L_a) + np.linalg.norm(L_b)) * (Output.T[n+1][0] - Output.T[n][0]) / 2

        L_a = L_b

    return effort


def AStar(nodes, n_start, n_goal):

    for key in nodes:
        nodes[key].heuristic = distanceCart(nodes[key], n_goal)

    O = [n_start]
    C = []
    n = 0

    while O[0] != n_goal and n < 10000:
        n += 1
        C.append(O[0])
        for key in O[0].neighbors:
            if nodes[key] not in C:
                p = nodes[key].heuristic + distanceCart(O[0], nodes[key]) + O[0].priority - O[0].heuristic
                if nodes[key] in O:
                    if p < nodes[key].priority:
                        nodes[key].priority = p
                        nodes[key].backpointer = O[0]
                else:
                    nodes[key].priority = p
                    nodes[key].backpointer = O[0]
                    O.append(nodes[key])
        O.pop(0)

        if not O:
            print("Dead end")
        else:
            O.sort(key = lambda x: x.priority)

    path = backtrack(O[0], n_start)

    return path


def effortBasedAStar(nodes, n_start, n_goal, omegaS, omegaG, avgOmega, I):

    O = [n_start]
    C = []
    n = 0

    while O[0] != n_goal and n < 10000:
        n += 1
        print(n)
        C.append(O[0])
        for key in O[0].neighbors:
            if nodes[key] not in C:
                path = backtrack(O[0], n_start)
                path.append(nodes[key])
                if nodes[key] != n_goal:
                    path.append(n_goal)
                Input = pathHandle(path, avgOmega)
                Output = spline(Input, omegaS, omegaG)
                p = effortEvaluation(Output, I)
                if nodes[key] in O:
                    if p < nodes[key].priority:
                        nodes[key].priority = p
                        nodes[key].backpointer = O[0]
                else:
                    nodes[key].priority = p
                    nodes[key].backpointer = O[0]
                    O.append(nodes[key])
        O.pop(0)

        if not O:
            print("Dead end")
        else:
            O.sort(key = lambda x: x.priority)

    path = backtrack(O[0], n_start)

    return path

# The following 'parametrize' function decorator provides the parameters and expected results for each
# of the multiple test runs for this test.
@pytest.mark.parametrize("N", [5,6])
@pytest.mark.parametrize("keepOutFov", [20])
@pytest.mark.parametrize("keepInFov", [70]) 
@pytest.mark.parametrize("costFcnType", [0,1])
@pytest.mark.parametrize("accuracy", [1e-12])
def test_constrainedAttitudeManeuver(show_plots, N, keepOutFov, keepInFov, costFcnType, accuracy):
    r"""
    **Validation Test Description**

    This unit test script tests the correctness of the path computed by the ConstrainedAttitudeManeuver module.
    The module is tested against Python scripts that mirror the same functions contained in the module. Tests are run for
    different grid coarseness levels, different keep-out fields of view, and one keep-in field of view.

    **Test Parameters**

    Args:
        N (int) : grid coarseness;
        keepOutFov (float) : Field of View (in radiants) of the keep-out boresight;
        keepInFov (float) : Field of View (in radiants) of the keep-in boresight;
        costFcnType (int) : 0 for the minimum MRP cartesian distance graph search, 1 for the effort-based graph search.
        accuracy (float): absolute accuracy value used in the validation tests

    **Description of Variables Being Tested**

    The tests to show the correctness of the module are the following:
    
    - First of all, an equivalent grid is built in python. The first test consists in comparing the nodes generated
      in Python versus the nodes generated in C++. The check consists in verifying whether the same key indices
      :math:`(i,j,k)` generate the same node coordinates :math:`\sigma_{BN}`. Secondly, it is checked whether 
      the same node is constraint-compliant or -incompliant both in Python and in C++.

    - After running the graph-search algorithm, a check is conduced to ensure the equivalence of the computed paths.
      Note that this unit test does not run the effort-based version of A*, due to the slow nature of the Python 
      implementation. If the user wishes, it is possible to uncomment line 429 to also test the effort-based
      graph-search algorithm.

    - The interpolated trajectory obtained in Python is checked versus the interpolated trajectory
      obtained in C++. The Python code uses the BSK-native :ref:`BSpline` library, which has its own unit test.

    - Lastly, a check is run on the norm of the required control torque for each time step of the
      interpolated trajectory.  The correctness of this check should imply the correctness of the
      functions used in the effort-based graph-search algorithm as well.
    """

    # each test method requires a single assert method to be called
    [testResults, testMessage] = CAMTestFunction(N, keepOutFov, keepInFov, costFcnType, accuracy)

    assert testResults < 1, testMessage

def CAMTestFunction(N, keepOutFov, keepInFov, costFcnType, accuracy):

    testFailCount = 0                       # zero unit test result counter
    testMessages = []                       # create empty array to store test log messages
    unitTaskName = "unitTask"               # arbitrary name (don't change)
    unitProcessName = "TestProcess"         # arbitrary name (don't change)

    Inertia = [0.02 / 3,  0.,         0.,
               0.,        0.1256 / 3, 0.,
               0.,        0.,         0.1256 / 3]
    InertiaTensor = unitTestSupport.np2EigenMatrix3d(Inertia)
    PlanetInertialPosition = np.array([10, 0, 0])
    SCInertialPosition = np.array([1, 0, 0])
    SCInitialAttitude = np.array([0, 0, -0.5])
    SCTargetAttitude = np.array([0, 0.5, 0])
    SCInitialAngRate = np.array([0, 0, 0])
    SCTargetAngRate = np.array([0, 0, 0])
    SCAvgAngRate = 0.03
    keepOutBoresight_B = [[1, 0, 0]]
    keepInBoresight_B = [[0, 1, 0], [0, 0, 1]]
    # convert Fov angles to radiants
    keepOutFov = keepOutFov * macros.D2R
    keepInFov = keepInFov * macros.D2R

    constraints = {'keepOut' : [], 'keepIn' : []}
    constraints['keepOut'].append( constraint(PlanetInertialPosition-SCInertialPosition, 'r') )
    constraints['keepIn'].append( constraint(PlanetInertialPosition-SCInertialPosition, 'g') )
    data =  {'keepOut_b' : keepOutBoresight_B, 'keepOut_fov' : [keepOutFov], 
             'keepIn_b' : keepInBoresight_B, 'keepIn_fov' : [keepInFov, keepInFov]}
    n_start = node(SCInitialAttitude, constraints, **data)
    n_goal  = node(SCTargetAttitude, constraints, **data)
    nodes = generateGrid(n_start, n_goal, N, constraints, data)
    if costFcnType == 0:
        path = AStar(nodes, n_start, n_goal)
    else:
        path = effortBasedAStar(nodes, n_start, n_goal, SCInitialAngRate, SCTargetAngRate, SCAvgAngRate, InertiaTensor)
    Input = pathHandle(path, SCAvgAngRate)
    Output = spline(Input, SCInitialAngRate, SCTargetAngRate)
    pathCost = effortEvaluation(Output, InertiaTensor)

    # Create a sim module as an empty container
    unitTestSim = SimulationBaseClass.SimBaseClass()

    # Create test thread
    simulationTime = macros.min2nano(2)
    testProcessRate = macros.sec2nano(0.5)     # update process rate update time
    testProc = unitTestSim.CreateNewProcess(unitProcessName)
    testProc.addTask(unitTestSim.CreateNewTask(unitTaskName, testProcessRate))
    
    testModule = constrainedAttitudeManeuver.ConstrainedAttitudeManeuver(N)
    testModule.sigma_BN_goal = SCTargetAttitude
    testModule.omega_BN_B_goal = SCTargetAngRate
    testModule.avgOmega = SCAvgAngRate
    testModule.BSplineType = 0
    testModule.costFcnType = costFcnType
    testModule.appendKeepOutDirection(keepOutBoresight_B[0], keepOutFov)
    testModule.appendKeepInDirection(keepInBoresight_B[0], keepInFov)
    testModule.appendKeepInDirection(keepInBoresight_B[1], keepInFov)
    testModule.ModelTag = "testModule"
    unitTestSim.AddModelToTask(unitTaskName, testModule)
	
    # connect messages
    SCStatesMsgData = messaging.SCStatesMsgPayload()
    SCStatesMsgData.r_BN_N = SCInertialPosition
    SCStatesMsgData.sigma_BN = SCInitialAttitude
    SCStatesMsgData.omega_BN_B = SCInitialAngRate
    SCStatesMsg = messaging.SCStatesMsg().write(SCStatesMsgData)
    VehicleConfigMsgData = messaging.VehicleConfigMsgPayload()
    VehicleConfigMsgData.ISCPntB_B = Inertia
    VehicleConfigMsg = messaging.VehicleConfigMsg().write(VehicleConfigMsgData)
    PlanetStateMsgData = messaging.SpicePlanetStateMsgPayload()
    PlanetStateMsgData.PositionVector = PlanetInertialPosition
    PlanetStateMsg = messaging.SpicePlanetStateMsg().write(PlanetStateMsgData)
    testModule.scStateInMsg.subscribeTo(SCStatesMsg)
    testModule.vehicleConfigInMsg.subscribeTo(VehicleConfigMsg)
    testModule.keepOutCelBodyInMsg.subscribeTo(PlanetStateMsg)
    testModule.keepInCelBodyInMsg.subscribeTo(PlanetStateMsg)

    numDataPoints = 200
    samplingTime = unitTestSupport.samplingTime(simulationTime, testProcessRate, numDataPoints)

    CAMLog = testModule.attRefOutMsg.recorder(samplingTime)
    unitTestSim.AddModelToTask(unitTaskName, CAMLog)
    CAMLogC = testModule.attRefOutMsgC.recorder(samplingTime)
    unitTestSim.AddModelToTask(unitTaskName, CAMLogC)

    # Need to call the self-init and cross-init methods
    unitTestSim.InitializeSimulation()

    # Set the simulation time.
    # NOTE: the total simulation time may be longer than this value. The
    # simulation is stopped at the next logging event on or after the
    # simulation end time.
    unitTestSim.ConfigureStopTime(simulationTime)        # seconds to stop simulation

    # Begin the simulation time run set above
    unitTestSim.ExecuteSimulation()

    timeData = CAMLog.times() * macros.NANO2SEC
    print(timeData)
    # print(len(CAMLog.sigma_RN))

    # check correctness of grid points:
    for i in range(-N,N+1):
        for j in range(-N,N+1):
            for k in range(-N,N+1):
                if (i, j, k) in nodes:
                    sigma_BN = nodes[(i, j, k)].sigma_BN
                    sigma_BN_BSK = []
                    for p in range(3):
                        sigma_BN_BSK.append( testModule.returnNodeCoord([i, j, k], p) )
                    if not unitTestSupport.isVectorEqual(sigma_BN, sigma_BN_BSK, accuracy):
                        testFailCount += 1
                        testMessages.append("FAILED: " + testModule.ModelTag + " Error in the coordinates of node ({},{},{}) \n".format(i, j, k))
                    if not nodes[(i, j, k)].isFree == testModule.returnNodeState([i, j, k]):
                        testFailCount += 1
                        testMessages.append("FAILED: " + testModule.ModelTag + " Error in the state of node ({},{},{}) \n".format(i, j, k))

    # check that the same path is produced
    for p in range(len(path)):
        sigma_BN = path[p].sigma_BN
        sigma_BN_BSK = []
        for j in range(3):
            sigma_BN_BSK.append(testModule.returnPathCoord(p,j))
        if not unitTestSupport.isVectorEqual(sigma_BN, sigma_BN_BSK, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in waypoint number {} in path \n".format(p))

    # check interpolated path compliance
    for n in range(len(Output.T)):
        T_BSK = testModule.Output.T[n][0]
        sigma_BSK = np.array([testModule.Output.X1[n][0], testModule.Output.X2[n][0], testModule.Output.X3[n][0]])
        sigmaDot_BSK = np.array([testModule.Output.XD1[n][0], testModule.Output.XD2[n][0], testModule.Output.XD3[n][0]])
        sigmaDDot_BSK = np.array([testModule.Output.XDD1[n][0], testModule.Output.XDD2[n][0], testModule.Output.XDD3[n][0]])

        T         = Output.T[n][0]
        sigma     = np.array([Output.X1[n][0], Output.X2[n][0], Output.X3[n][0]])
        sigmaDot  = np.array([Output.XD1[n][0], Output.XD2[n][0], Output.XD3[n][0]])
        sigmaDDot = np.array([Output.XDD1[n][0], Output.XDD2[n][0], Output.XDD3[n][0]])

        if not unitTestSupport.isDoubleEqual(T, T_BSK, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in time at index #{} in trajectory \n".format(n))
        if not unitTestSupport.isVectorEqual(sigma, sigma_BSK, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in sigma at index #{} in trajectory \n".format(n))
        if not unitTestSupport.isVectorEqual(sigmaDot, sigmaDot_BSK, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in sigmaDot at index #{} in trajectory \n".format(n))
        if not unitTestSupport.isVectorEqual(sigmaDDot, sigmaDDot_BSK, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in sigmaDDot at index #{} in trajectory \n".format(n))

    # check same path cost for every spline point
    for n in range(len(Output.T)):
        c1 = testModule.computeTorqueNorm(n, Inertia)

        sigma     = [Output.X1[n][0], Output.X2[n][0], Output.X3[n][0]]
        sigmaDot  = [Output.XD1[n][0], Output.XD2[n][0], Output.XD3[n][0]]
        sigmaDDot = [Output.XDD1[n][0], Output.XDD2[n][0], Output.XDD3[n][0]]
        c2 = np.linalg.norm(computeTorque(sigma, sigmaDot, sigmaDDot, InertiaTensor))

        if not unitTestSupport.isDoubleEqual(c1, c2, accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error torque norm in point {} in trajectory \n".format(n))

    if not unitTestSupport.isDoubleEqual(pathCost, testModule.pathCost, accuracy):
        testFailCount += 1
        testMessages.append("FAILED: " + testModule.ModelTag + " Error in path cost \n")

    # check correctness of output message
    for n in range(len(timeData)):
        t = timeData[n]
        sigma_RN     = [Output.getStates(t,0,0), Output.getStates(t,0,1), Output.getStates(t,0,2)]
        sigmaDot_RN  = [Output.getStates(t,1,0), Output.getStates(t,1,1), Output.getStates(t,1,2)]
        sigmaDDot_RN = [Output.getStates(t,2,0), Output.getStates(t,2,1), Output.getStates(t,2,2)]
        RN = rbk.MRP2C(sigma_RN)
        omega_RN_R = rbk.dMRP2Omega(sigma_RN, sigmaDot_RN)
        omega_RN_N = np.matmul(omega_RN_R, RN)
        omegaDot_RN_R = rbk.ddMRP2dOmega(sigma_RN, sigmaDot_RN, sigmaDDot_RN)
        omegaDot_RN_N = np.matmul(omegaDot_RN_R, RN)

        if not unitTestSupport.isVectorEqual(sigma_RN, CAMLog.sigma_RN[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in attitude reference message at t = {} sec \n".format(t))
        if not unitTestSupport.isVectorEqual(omega_RN_N, CAMLog.omega_RN_N[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in attitude reference message at t = {} sec \n".format(t))
        if not unitTestSupport.isVectorEqual(omegaDot_RN_N, CAMLog.domega_RN_N[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in attitude reference message at t = {} sec \n".format(t))

        if not unitTestSupport.isVectorEqual(sigma_RN, CAMLogC.sigma_RN[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in C attitude reference message at t = {} sec \n".format(t))
        if not unitTestSupport.isVectorEqual(omega_RN_N, CAMLogC.omega_RN_N[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in C attitude reference message at t = {} sec \n".format(t))
        if not unitTestSupport.isVectorEqual(omegaDot_RN_N, CAMLogC.domega_RN_N[n], accuracy):
            testFailCount += 1
            testMessages.append("FAILED: " + testModule.ModelTag + " Error in C attitude reference message at t = {} sec \n".format(t)) 


    return [testFailCount, ''.join(testMessages)]
	   

#
# This statement below ensures that the unitTestScript can be run as a
# stand-along python script
#
if __name__ == "__main__":
    CAMTestFunction(
        5,      # grid coarsness N
        20,      # keepOutFov
        70,      # keepInFov
        1,       # costFcnType
        1e-12    # accuracy
        )