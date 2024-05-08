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
from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import macros as mc

np.set_printoptions(precision=12)

def printResults3DSpin(sigma_R0N, omega_R0N_N, domega_R0N_N, omega_RR0_R, dt):

    # Compute angular Rate
    RN = rbk.MRP2C(sigma_R0N)
    omega_RR0_N = np.dot(RN.T, omega_RR0_R)
    omega_RN_N = omega_RR0_N + omega_R0N_N

    # Compute angular acceleration
    domega_RN_N = np.cross(omega_R0N_N, omega_RR0_N) + domega_R0N_N

    # Compute attitude
    omega_RN_R = np.dot(RN, omega_RN_N)
    B = rbk.BmatMRP(sigma_R0N)
    dsigma_RN = 0.25 * np.dot(B, omega_RN_R)
    sigma_RN =  sigma_R0N + dsigma_RN * dt
    rbk.MRPswitch(sigma_RN, 1)

    # Print results
    print('sigma_RN = ', sigma_RN)
    print('omega_RN_N = ', omega_RN_N)
    print('domega_RN_N = ', domega_RN_N)
    print('\n')
    return sigma_RN


sigma_R0N = np.array([0.1, 0.2, 0.3])
omega_R0N_N = np.array([0., 0., 0.])
domega_R0N_N = np.array([0., 0., 0.])
omega_spin = np.array([1., -1., 0.5]) * mc.D2R
print('CallTime = 0.0')
dt = 0.0
sigma_RN = printResults3DSpin(sigma_R0N, omega_R0N_N, domega_R0N_N, omega_spin, dt)
dt = 0.5
sigma_RN = printResults3DSpin(sigma_RN, omega_R0N_N, domega_R0N_N, omega_spin, dt)
dt = 0.5
printResults3DSpin(sigma_RN, omega_R0N_N, domega_R0N_N, omega_spin, dt)