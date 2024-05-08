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
import numpy as np
from Basilisk.fswAlgorithms import rwMotorTorque
from numpy import linalg as la


def controlAxes3D():
    C = np.array([
        [1.0, 0.0, 0.0]
        , [0.0, 1.0, 0.0]
        , [0.0, 0.0, 1.0]
    ])
    return C
def controlAxes2D():
    C = np.array([
        [1.0, 0.0, 0.0]
        , [0.0, 0.0, 1.0]
    ])
    return C
def controlAxes1D():
    C = np.array([
        [1.0, 0.0, 0.0]
    ])
    return C



def computeTorqueU(CArray, Gs_B, Lr, availMsg):

    numControlAxes = len(CArray)//3
    numWheels = len(availMsg)
    nonAvailWheels = 0

    # Build Control Frame (doesn't need to be a complete frame)
    C = np.zeros((3,3))
    for i in range(3):
        if numControlAxes > i:
            C[i,:] = CArray[3*i:3*(i+1)]
        else:
            C[i,:] = [0.0, 0.0, 0.0]

    # Remove wheels that are deemed unavailable
    for i in range(len(Gs_B[0])): #
        if numWheels > i:
            if availMsg[i] is not rwMotorTorque.AVAILABLE:
                Gs_B[:,i] = [0.0, 0.0, 0.0]
                nonAvailWheels += 1
        else:
            Gs_B[:,i] = [0.0, 0.0, 0.0]

    # If fewer wheels than number of control axes, output no torque
    if (numWheels-nonAvailWheels) < numControlAxes:
        return [0.0]*len(Gs_B[0])


    Lr_C = np.dot(C,Lr) # Project torque onto control axes
    CGs = np.dot(C, Gs_B) # Map the control axes onto the wheels

    # Build minimum norm framework
    M = np.dot(CGs, CGs.T)
    M_rep = np.identity(3) # Need to keep the matrix non-singular for inversion
    for i in range(0,numControlAxes):
        for j in range(0,numControlAxes):
            M_rep[i][j] = M[i][j]
    M_inv = la.inv(M_rep)

    # Remove projection to any non-defined control axes
    for i in range(numControlAxes,3):
        M_inv[i][i] = 0.0

    # Determine the solution
    v3_temp = np.dot(M_inv, Lr_C)

    # Map the solution to the wheels
    u_s = np.dot(CGs.T, v3_temp)

    return -u_s

def exampleComputation():
    Gs_B = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5773502691896258, 0.5773502691896258, 0.5773502691896258]
    ]).T

    JsList = np.array([0.1, 0.1, 0.1, 0.1])
    numRW = 4
    rwConfigParams = (Gs_B, JsList, numRW)

    Lr = np.array([1.0, -0.5, 0.7])
    rwAvailability = np.array([1, 1, 1, 1])

    print('3D Control')
    u_s = computeTorqueU(controlAxes3D(), Gs_B, Lr)
    print('U_s = ', u_s, '\n')

    print('2D Control')
    u_s = computeTorqueU(controlAxes2D(), Gs_B, Lr)
    print('U_s = ', u_s)

    print('1D Control')
    u_s = computeTorqueU(controlAxes1D(), Gs_B, Lr)
    print('U_s = ', u_s)
