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
from numpy import sin, cos
np.set_printoptions(precision=12)

from Basilisk.utilities import RigidBodyKinematics as rbk
from Basilisk.utilities import astroFunctions as af

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def printResults_VelocityPoint(r_BN_N, v_BN_N, celBodyPosVec, celBodyVelVec, mu):
    r = r_BN_N - celBodyPosVec
    v = v_BN_N - celBodyVelVec

    h = np.cross(r, v)
    i_r = af.normalize(r)
    i_v = normalize(v)
    i_h = normalize(h)
    i_n = np.cross(i_v, i_h)
    VN = np.array([ i_n, i_v, i_h ])
    sigma_VN = rbk.C2MRP(VN)

    hm = la.norm(h)
    rm = la.norm(r)
    drdt = np.dot(v, i_r)
    dfdt = hm / (rm * rm)
    ddfdt2 = -2.0 * drdt / rm * dfdt

    (a, e, i, Omega, omega, f) = af.RV2OE(mu, r, v)
    den = 1 + e * e + 2 * e * cos(f)
    temp = e * (e + cos(f)) / den
    dBdt = temp * dfdt
    ddBdt2 = temp * ddfdt2 + (e * (e * e - 1) * sin(f)) / (den * den) * dfdt * dfdt

    omega_VN_N = (-dBdt + dfdt) * i_h
    domega_VN_N = (-ddBdt2 + ddfdt2) * i_h

    print('sigma_VN = ', sigma_VN)
    print('omega_VN_N = ', omega_VN_N)
    print('domega_VN_N = ', domega_VN_N)

    return (sigma_VN, omega_VN_N, domega_VN_N)

# MAIN
# Initial Conditions (IC)
a = af.E_radius * 2.8
e = 0.8
i = 0.0
Omega = 0.0
omega = 0.0
f = 60 * af.D2R
(r, v) = af.OE2RV(af.mu_E, a, e, i, Omega, omega, f)
r_BN_N = r
v_BN_N = v
celBodyPosVec = np.array([0.0, 0.0, 0.0])
celBodyVelVec = np.array([0.0, 0.0, 0.0])
# Print generated Velocity Frame for the given IC
printResults_VelocityPoint(r_BN_N, v_BN_N, celBodyPosVec, celBodyVelVec, af.mu_E)