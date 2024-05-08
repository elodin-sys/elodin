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
from numpy import linalg as la
np.set_printoptions(precision=12)





from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import astroFunctions as af

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def printResults_HillPoint(r_BN_N, v_BN_N, celBodyPosVec, celBodyVelVec):
    r = r_BN_N - celBodyPosVec
    v = v_BN_N - celBodyVelVec
    h = np.cross(r, v)
    i_r = normalize(r)
    i_h = normalize(h)
    i_theta = np.cross(i_h, i_r)
    HN = np.array([ i_r, i_theta, i_h ])
    sigma_HN = rbk.C2MRP(HN)

    hm = la.norm(h)
    rm = la.norm(r)
    drdt = np.dot(v, i_r)
    dfdt = hm / (rm * rm)
    ddfdt2 = -2.0 * drdt / rm * dfdt

    omega_HN_N = dfdt * i_h
    domega_HN_N = ddfdt2 * i_h

    print('sigma_HN = ', sigma_HN)
    print('omega_HN_N = ', omega_HN_N)
    print('domega_HN_N = ', domega_HN_N)

    HN = rbk.MRP2C(sigma_HN)
    M = rbk.Mi(0.5*np.pi, 1)
    sigma = rbk.C2MRP(np.dot(M, HN))
    print(sigma)
    return (sigma_HN, omega_HN_N, domega_HN_N)

# MAIN
# Initial Conditions (IC)
a = af.E_radius * 2.8
e = 0.0
i = 0.0
Omega = 0.0
omega = 0.0
f = 60 * af.D2R
(r, v) = af.OE2RV(af.mu_E, a, e, i, Omega, omega, f)
r_BN_N = r
v_BN_N = v
celBodyPosVec = np.array([0.0, 0.0, 0.0])
celBodyVelVec = np.array([0.0, 0.0, 0.0])
# Print generated Hill Frame for the given IC
printResults_HillPoint(r_BN_N, v_BN_N, celBodyPosVec, celBodyVelVec)
