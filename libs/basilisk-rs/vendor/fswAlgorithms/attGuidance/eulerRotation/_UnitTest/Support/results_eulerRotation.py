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
from numpy import sin, cos

np.set_printoptions(precision=12)

from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros as mc




def printResults_eulerRotation(configData, sigma_R0N, omega_R0N_N, domega_R0N_N, callTime):
    def computeEuler321_Binv_derivative(angleSet, angleRates):
        theta = angleSet[1]
        phi = angleSet[2]
        thetaDot = angleRates[1]
        phiDot = angleRates[2]
        B_inv_deriv = [
            [-thetaDot*cos(theta), 0, 0],
            [phiDot*cos(phi)*cos(theta)-thetaDot*sin(phi)*sin(theta), -phiDot*sin(phi), 0],
            [-phiDot*sin(phi)*cos(theta)-thetaDot*cos(phi)*cos(theta), -phiDot*cos(phi), 0]
        ]
        return B_inv_deriv

    angleSet = configData[0]
    angleRates = configData[1]
    dt = configData[2]

    # Compute attitude set
    R0N = rbk.MRP2C(sigma_R0N)
    angleSet += dt * angleRates
    RR0 = rbk.euler3212C(angleSet)
    RN = np.dot(RR0, R0N)
    sigma_RN = rbk.C2MRP(RN)

    # Compute angular velocity
    B_inv = rbk.BinvEuler321(angleSet)
    omega_RR0_R = np.dot(B_inv, angleRates)
    omega_RR0_N = np.dot(RN.T, omega_RR0_R)
    omega_RN_N = omega_RR0_N + omega_R0N_N

    # Compute angular acceleration
    B_inv_deriv = computeEuler321_Binv_derivative(angleSet, angleRates)
    domega_RR0_R = np.dot(B_inv_deriv, angleRates)
    domega_RR0_N = np.dot(RN.T, domega_RR0_R) + np.cross(omega_R0N_N, omega_RR0_N)
    domega_RN_N = domega_RR0_N + domega_R0N_N

    # Print results
    def printData():
        print('callTime = ', callTime)
        print('eulerAngleSet = ', angleSet)
        print('B_inv_deriv = ', B_inv_deriv)
        print('sigma_RN = ', sigma_RN)
        print('omega_RN_N = ', omega_RN_N)
        print('domega_RN_N = ', domega_RN_N)
        print('\n')
    printData()
    return angleSet


# Initial Conditions
sigma_R0N = np.array([ 0.1, 0.2, 0.3 ])
omega_R0N_N = np.array([0.1, 0.0, 0.0])
domega_R0N_N = np.array([0.0, 0.0, 0.0])

dt = 0.5
angleRates = np.array([0.1, 0., 0.]) * mc.D2R

angleSet = np.array([0.0, 0.0, 0.0]) * mc.D2R
configData = (angleSet, angleRates, 0.0)
angleSet = printResults_eulerRotation(configData, sigma_R0N, omega_R0N_N, domega_R0N_N, dt*0.0)
configData = (angleSet, angleRates, dt)
angleSet = printResults_eulerRotation(configData, sigma_R0N, omega_R0N_N, domega_R0N_N, dt*1.0)
configData = (angleSet, angleRates, dt)
angleSet = printResults_eulerRotation(configData, sigma_R0N, omega_R0N_N, domega_R0N_N, dt*2.0)


# t0 = 0.0
# t1 = 2000
# span = (t1 - t0)/dt + 1
# t_vec = np.linspace(t0, t1, int(span))
# psi_vec = np.array([])
# theta_vec = np.array([])
# phi_vec = np.array([])
#
# rx_vec = np.array([])
# ry_vec = np.array([])
# rz_vec = np.array([])
# for t in t_vec:
#     angleSet = printResults_eulerRotation(configData, sigma_R0N, omega_R0N_N, domega_R0N_N, t)
#     configData = (angleSet, angleRates, dt)
#     psi_vec = np.append(psi_vec, angleSet[0])
#     theta_vec = np.append(theta_vec, angleSet[1])
#     phi_vec = np.append(phi_vec, angleSet[2])
#
#     rx = cos(angleSet[1]) * cos(angleSet[0])
#     ry = cos(angleSet[1]) * sin(angleSet[0])
#     rz = sin(angleSet[1])
#     rx_vec = np.append(rx_vec, rx)
#     ry_vec = np.append(ry_vec, ry)
#     rz_vec = np.append(rz_vec, rz)
#
# print 'rx_vec = ', rx_vec
# print 'ry_vec = ', ry_vec
# print 'rz_vec = ', rz_vec
#
# def plotEuler321():
#     plt.figure(1)
#     plt.plot(t_vec, psi_vec, t_vec, theta_vec, t_vec, phi_vec)
#     plt.legend(['$\psi$','$\Theta$', '$\phi$'])
# def plotPlanes2D():
#     plt.figure(2)
#     plt.plot(rx_vec, rz_vec)
#     plt.xlabel('$R_X$')
#     plt.ylabel('$R_Z$')
#     plt.title('XZ')
#     plt.figure(3)
#     plt.plot(rx_vec, ry_vec)
#     plt.xlabel('$R_X$')
#     plt.ylabel('$R_Y$')
#     plt.title('XY')
#     plt.figure(4)
#     plt.plot(ry_vec, rz_vec)
#     plt.xlabel('$R_Y$')
#     plt.ylabel('$R_Z$')
#     plt.title('YZ')
# def plot3D():
#     fig = plt.figure(10)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(rx_vec, ry_vec, rz_vec)
#     max_range = np.array([rx_vec.max() - rx_vec.min(), ry_vec.max() - ry_vec.min(), rz_vec.max() - rz_vec.min()]).max()
#     Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (rx_vec.max() + rx_vec.min())
#     Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ry_vec.max() + ry_vec.min())
#     Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (rz_vec.max() + rz_vec.min())
#     # Comment or uncomment following both lines to test the fake bounding box:
#     for xb, yb, zb in zip(Xb, Yb, Zb):
#         ax.plot([xb], [yb], [zb], 'w')
#
# plotEuler321()
# plotPlanes2D()
# plot3D()
# plt.show()


# numPoints = 36
# totalSimTime = 60 * 20 * 4
# angleSetList = np.zeros(3*numPoints)
# angleRatesList = []
# rasterTimeList = np.zeros(numPoints)
# for i in range(numPoints):
#     angleSetList[3 * i + 1] = 2 * np.pi * i / numPoints
#     rasterTimeList[i] = totalSimTime / numPoints
# print angleSetList
#
# M3 = rbk.Mi(np.pi, 3)
# print 'M3 = ', M3
# print 'sigma = ', rbk.C2MRP(M3)




