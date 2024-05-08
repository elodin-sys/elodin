/*
 ISC License

 Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

 Permission to use, copy, modify, and/or distribute this software for any
 purpose with or without fee is hereby granted, provided that the above
 copyright notice and this permission notice appear in all copies.

 THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

 */

#include "orbitalMotion.h"

#include <math.h>
#include <stdio.h>
#include <stdarg.h>

#include "linearAlgebra.h"
#include "astroConstants.h"
#include "architecture/utilities/bsk_Print.h"
#include "rigidBodyKinematics.h"


/*!
 * Purpose: maps inertial position and velocity vectors in the Hill frame DCM HN
 * Inputs:
 *   rc_N: inertial position vector
 *   vc_N: inertial velocity vector
 * Outputs:
 *   HN: Hill frame DCM relative to inertial frame
 */
void hillFrame(double *rc_N, double *vc_N, double HN[3][3])
{
    double ir_N[3];         /* orbit radial unit direction vector */
    double itheta_N[3];     /* along-track unit direction vector */
    double ih_N[3];         /* orbit plane normal unit direction vector */
    double hVec_N[3];       /* orbit angular momentum vector */

    v3Normalize(rc_N, ir_N);
    v3Cross(rc_N, vc_N, hVec_N);
    v3Normalize(hVec_N, ih_N);
    v3Cross(ih_N, ir_N, itheta_N);
    v3Copy(ir_N, HN[0]);
    v3Copy(itheta_N, HN[1]);
    v3Copy(ih_N, HN[2]);

    return;
}

/*!
 * Purpose: maps Hill frame deputy states to inertial inertial position and velocity vectors
 * Inputs:
 *   rc_N: chief inertial position vector
 *   vc_N: chief inertial velocity vector
 *   rho_H: deputy Hill position vector
 *   rhoPrime_H: deputy Hill velocity vector
 * Outputs:
 *   rd_N: deputy inertial position vector
 *   vd_N: deputy inertial velocity vector
 */
void  hill2rv(double *rc_N, double *vc_N, double *rho_H, double *rhoPrime_H, double *rd_N, double *vd_N)
{
    double HN[3][3];        /* DCM of Hill frame relative to inertial */
    double NH[3][3];        /* DCM of inertial frame relative to Hill */
    double hVec_N[3];       /* orbit angular momentum vector */
    double rc;              /* chief orbit radius */
    double fDot;            /* chief true anomaly rate */
    double omega_HN_H[3];   /* Hill frame angular velocity */
    double rho_N[3];        /* deputy relative to chief vector in N frame components */

    hillFrame(rc_N, vc_N, HN);
    m33Transpose(HN, NH);
    v3Cross(rc_N, vc_N, hVec_N);
    rc = v3Norm(rc_N);
    fDot = v3Norm(hVec_N)/rc/rc;
    v3Set(0, 0, fDot, omega_HN_H);

    /* compute inertial deputy position */
    m33MultV3(NH, rho_H, rho_N);
    v3Add(rc_N, rho_N, rd_N);

    /* compute inertial deputy velocity */
    v3Cross(omega_HN_H, rho_H, vd_N);
    v3Add(vd_N, rhoPrime_H, vd_N);
    m33MultV3(NH, vd_N, vd_N);
    v3Add(vd_N, vc_N, vd_N);

    return;
}


/*!
 * Purpose: maps inertial frame deputy states to Hill inertial position and velocity vectors
 * Inputs:
 *   rc_N: chief inertial position vector
 *   vc_N: chief inertial velocity vector
 *   rd_N: deputy inertial position vector
 *   vd_N: deputy inertial velocity vector
 * Outputs:
 *   rho_H: deputy Hill position vector
 *   rhoPrime_H: deputy Hill velocity vector
 */
void    rv2hill(double *rc_N, double *vc_N, double *rd_N, double *vd_N, double *rho_H, double *rhoPrime_H)
{
    double HN[3][3];        /* DCM of Hill frame relative to inertial */
    double hVec_N[3];       /* orbit angular momentum vector */
    double rc;              /* chief orbit radius */
    double fDot;            /* chief true anomaly rate */
    double omega_HN_H[3];   /* Hill frame angular velocity */
    double rho_N[3];        /* deputy relative to chief vector in N frame components */
    double rhoDot_N[3];     /* inertial derivative of deputy/chief vector */
    double rhoDot_H[3];     /* inertial derivative of deputy/chief vector */

    hillFrame(rc_N, vc_N, HN);
    v3Cross(rc_N, vc_N, hVec_N);
    rc = v3Norm(rc_N);
    fDot = v3Norm(hVec_N)/rc/rc;
    v3Set(0, 0, fDot, omega_HN_H);

    /* compute Hill frame position */
    v3Subtract(rd_N, rc_N, rho_N);
    m33MultV3(HN, rho_N, rho_H);

    /* compute Hill frame velocity */
    v3Subtract(vd_N, vc_N, rhoDot_N);
    m33MultV3(HN, rhoDot_N, rhoDot_H);
    v3Cross(omega_HN_H, rho_H, rhoPrime_H);
    v3Subtract(rhoDot_H, rhoPrime_H, rhoPrime_H);
    
    return;
}


/*!
 * Purpose: Maps eccentric anomaly angles into true anomaly angles.
 *   This function requires the orbit to be either circular or
 *   non-rectilinar elliptic orbit.
 * Inputs:
 *   Ecc = eccentric anomaly (rad)
 *   e = eccentricity (0 <= e < 1)
 * Outputs:
 *   f = true anomaly (rad)
 */
double E2f(double Ecc, double e)
{
    double f;

    if((e >= 0) && (e < 1)) {
        f = 2 * atan2(sqrt(1 + e) * sin(Ecc / 2), sqrt(1 - e) * cos(Ecc / 2));
    } else {
        f = NAN;
        BSK_PRINT(MSG_ERROR, "E2f() received e = %g. The value of e should be 0 <= e < 1.", e);
    }

    return f;
}

/*!
 * Purpose: Maps the eccentric anomaly angle into the corresponding
 *   mean elliptic anomaly angle.  Both 2D and 1D elliptic
 *   orbit are allowed.
 * Inputs:
 *   Ecc = eccentric anomaly (rad)
 *   e = eccentricity (0 <= e < 1)
 * Outputs:
 *   M = mean elliptic anomaly (rad)
 */
double E2M(double Ecc, double e)
{
    double M;

    if((e >= 0) && (e < 1)) {
        M = Ecc - e * sin(Ecc);
    } else {
        M = NAN;
        BSK_PRINT(MSG_ERROR, "E2M() received e = %g. The value of e should be 0 <= e < 1.", e);
    }

    return M;
}

/*!
 * Purpose: Maps true anomaly angles into eccentric anomaly angles.
 *   This function requires the orbit to be either circular or
 *   non-rectilinar elliptic orbit.
 * Inputs:
 *   f = true anomaly angle (rad)
 *   e = eccentricity (0 <= e < 1)
 * Outputs:
 *   Ecc = eccentric anomaly (rad)
 */
double f2E(double f, double e)
{
    double Ecc;

    if((e >= 0) && (e < 1)) {
        Ecc = 2 * atan2(sqrt(1 - e) * sin(f / 2), sqrt(1 + e) * cos(f / 2));
    } else {
        Ecc = NAN;
        BSK_PRINT(MSG_ERROR, "f2E() received e = %g. The value of e should be 0 <= e < 1.", e);
    }

    return Ecc;
}

/*!
 * Purpose: Maps true anomaly angles into hyperbolic anomaly angles.
 *   This function requires the orbit to be hyperbolic
 * Inputs:
 *   f = true anomaly angle (rad)
 *   e = eccentricity (e > 1)
 * Outputs:
 *   H = hyperbolic anomaly (rad)
 */
double f2H(double f, double e)
{
    double H;

    if(e > 1) {
        H = 2 * atanh(sqrt((e - 1) / (e + 1)) * tan(f / 2));
    } else {
        H = NAN;
        BSK_PRINT(MSG_ERROR, "f2H() received e = %g. The value of e should be 1 < e.", e);
    }

    return H;
}

/*!
 * Purpose: Maps hyperbolic anomaly angles into true anomaly angles.
 *   This function requires the orbit to be hyperbolic
 * Inputs:
 *   H = hyperbolic anomaly (rad)
 *   e = eccentricity (e > 1)
 * Outputs:
 *   f = true anomaly angle (rad)
 */
double H2f(double H, double e)
{
    double f;

    if(e > 1) {
        f = 2 * atan(sqrt((e + 1) / (e - 1)) * tanh(H / 2));
    } else {
        f = NAN;
        BSK_PRINT(MSG_ERROR, "H2f() received e = %g. The value of e should be 1 < e.", e);
    }

    return f;
}

/*!
 * Purpose: Maps the hyperbolic anomaly angle H into the corresponding
 *   mean hyperbolic anomaly angle N.
 * Inputs:
 *   H = hyperbolic anomaly (rad)
 *   e = eccentricity (e > 1)
 * Outputs:
 *   N = mean hyperbolic anomaly (rad)
 */
double H2N(double H, double e)
{
    double N;

    if(e > 1) {
        N = e * sinh(H) - H;
    } else {
        N = NAN;
        BSK_PRINT(MSG_ERROR, "H2N() received e = %g. The value of e should be 1 < e.", e);
    }

    return N;
}

/*!
 * Purpose: Maps the mean elliptic anomaly angle into the corresponding
 *   eccentric anomaly angle.  Both 2D and 1D elliptic
 *   orbit are allowed.
 * Inputs:
 *   M = mean elliptic anomaly (rad)
 *   e = eccentricity (0 <= e < 1)
 * Outputs:
 *   Ecc = eccentric anomaly (rad)
 */
double M2E(double M, double e)
{
    double small = 1e-13;
    double dE = 10 * small;
    double E1 = M;
    int    max = 200;
    int    count = 0;

    if((e >= 0) && (e < 1)) {
        while(fabs(dE) > small) {
            dE = (E1 - e * sin(E1) - M) / (1 - e * cos(E1));
            E1 -= dE;
            if(++count > max) {
                BSK_PRINT(MSG_ERROR, "iteration error in M2E(%f,%f)", M, e);
                dE = 0.;
            }
        }
    } else {
        E1 = NAN;
        BSK_PRINT(MSG_ERROR, "M2E() received e = %g. The value of e should be 0 <= e < 1.", e);
    }

    return E1;
}

/*!
 * Purpose: Maps the mean hyperbolic anomaly angle N into the corresponding
 *   hyperbolic anomaly angle H.
 * Inputs:
 *   N = mean hyperbolic anomaly (rad)
 *   e = eccentricity (e > 1)
 * Outputs:
 *   H = hyperbolic anomaly (rad)
 */
double N2H(double N, double e)
{
    double small = 1e-13;
    double dH = 10 * small;
    double H1 = N;
    int    max = 200;
    int    count = 0;
    if(fabs(H1) > 7.0)
    {
        H1 = N/fabs(N)*7.0;
    }

    if(e > 1) {
        while(fabs(dH) > small) {
            dH = (e * sinh(H1) - H1 - N) / (e * cosh(H1) - 1);
            H1 -= dH;
            if(++count > max) {
                BSK_PRINT(MSG_ERROR, "iteration error in N2H(%f,%f)", N, e);
                dH = 0.;
            }
        }
    } else {
        H1 = NAN;
        BSK_PRINT(MSG_ERROR, "N2H() received e = %g. The value of e should be e > 1.", e);
    }

    return H1;
}

/*!
 * Purpose: Translates the orbit elements
 *           a   - semi-major axis           (km)
 *           e   - eccentricity
 *           i   - inclination               (rad)
 *           AN  - ascending node            (rad)
 *           AP  - argument of periapses     (rad)
 *           f   - true anomaly angle        (rad)
 *   to the inertial Cartesian position and velocity vectors.
 *   The attracting body is specified through the supplied
 *   gravitational constant mu (units of km^3/s^2).
 *
 *   The code can handle the following cases:
 *      circular:       e = 0           a > 0
 *      elliptical-2D:  0 < e < 1       a > 0
 *      elliptical-1D:  e = 1           a > 0        f = Ecc. Anom. here
 *      parabolic:      e = 1           rp = -a
 *      hyperbolic:     e > 1           a < 0
 *
 *   Note: to handle the parabolic case and distinguish it form the
 *   rectilinear elliptical case, instead of passing along the
 *   semi-major axis a in the "a" input slot, the negative radius
 *   at periapses is supplied.  Having "a" be negative and e = 1
 *   is a then a unique identified for the code for the parabolic
 *   case.
 * Inputs:
 *   mu = gravitational parameter
 *   elements = orbital elements
 * Outputs:
 *   rVec = position vector
 *   vVec = velocity vector
 */
void elem2rv(double mu, classicElements *elements, double *rVec, double *vVec)
{
    double e;                   /* eccentricty */
    double a;                   /* semi-major axis */
    double f;                   /* true anomaly */
    double r;                   /* orbit radius */
    double v;                   /* orbit velocity magnitude */
    double i;                   /* orbit inclination angle */
    double p;                   /* the parameter or the semi-latus rectum */
    double AP;                  /* argument of perigee */
    double AN;                  /* argument of the ascending node */
    double theta;               /* true latitude theta = omega + f */
    double h;                   /* orbit angular momentum magnitude */
    double ir[3];               /* orbit radius unit vector */
    double eps;                 /* small numerical value parameter */

    /* define what is a small numerical value */
    eps = 1e-12;

    /* map classical elements structure into local variables */
    a = elements->a;
    e = elements->e;
    i = elements->i;
    AN = elements->Omega;
    AP = elements->omega;
    f = elements->f;

    if((fabs(e-1.0) < eps) && (fabs(a) > eps)) { /* 1D rectilinear elliptic/hyperbolic orbit case */

        double angle;                 /* eccentric/hyperbolic anomaly */
        angle = f;                    /* f is treated as ecc/hyp anomaly */
        /* orbit radius  */
        if (elements->a > 0.0) {
            r = a * (1 - e * cos(angle));
        } else {
            r = a * (1 - e * cosh(angle));
        }
        v = sqrt(2 * mu / r - mu / a);
        ir[0] = cos(AN) * cos(AP) - sin(AN) * sin(AP) * cos(i);
        ir[1] = sin(AN) * cos(AP) + cos(AN) * sin(AP) * cos(i);
        ir[2] = sin(AP) * sin(i);
        v3Scale(r, ir, rVec);
        if(sin(angle) > 0) {
            v3Scale(v, ir, vVec);
        } else {
            v3Scale(-v, ir, vVec);
        }
    } else { /* general 2D orbit case */
        /* evaluate semi-latus rectum */
        if(fabs(a) > eps) {
            p = a * (1 - e * e);        /* elliptic or hyperbolic */
        } else {
            p = 2 * elements->rPeriap;  /* parabolic */
        }

        r = p / (1 + e * cos(f));   /* orbit radius */
        theta = AP + f;             /* true latitude angle */
        h = sqrt(mu * p);           /* orbit ang. momentum mag. */

        rVec[0] = r * (cos(AN) * cos(theta) - sin(AN) * sin(theta) * cos(i));
        rVec[1] = r * (sin(AN) * cos(theta) + cos(AN) * sin(theta) * cos(i));
        rVec[2] = r * (sin(theta) * sin(i));

        vVec[0] = -mu / h * (cos(AN) * (sin(theta) + e * sin(AP)) + sin(AN) * (cos(
                                 theta) + e * cos(AP)) * cos(i));
        vVec[1] = -mu / h * (sin(AN) * (sin(theta) + e * sin(AP)) - cos(AN) * (cos(
                                 theta) + e * cos(AP)) * cos(i));
        vVec[2] = -mu / h * (-(cos(theta) + e * cos(AP)) * sin(i));
    }
}


/*!
 * Purpose: Translates the orbit elements inertial Cartesian position
 *   vector rVec and velocity vector vVec into the corresponding
 *   classical orbit elements where
 *           a   - semi-major axis (zero if parabolic)
 *           e   - eccentricity
 *           i   - inclination               (rad)
 *           AN  - ascending node            (rad)
 *           AP  - argument of periapses     (rad)
 *           f   - true anomaly angle        (rad)
 *                 if the orbit is rectilinear, then this will be the
 *                 eccentric or hyperbolic anomaly
 *           rp  - radius at periapses
 *           ra  - radius at apoapses (zero if parabolic)
 *   The attracting body is specified through the supplied
 *   gravitational constant mu (units of km^3/s^2).
 * Inputs:
 *   mu = gravitational parameter
 *   rVec = position vector
 *   vVec = velocity vector
 * Outputs:
 *   elements = orbital elements
 */
void rv2elem(double mu, double *rVec, double *vVec, classicElements *elements)
{
    double hVec[3];             /* orbit angular momentum vector */
    double ihHat[3];            /* normalized orbit angular momentum vector */
    double h;                   /* orbit angular momentum magnitude */
    double v3[3];               /* temp vector */
    double n1Hat[3];            /* 1st inertial frame base vector */
    double n3Hat[3];            /* 3rd inertial frame base vector */
    double nVec[3];             /* line of nodes vector */
    double inHat[3];            /* normalized line of nodes vector */
    double irHat[3];            /* normalized position vector */
    double r;                   /* current orbit radius */
    double v;                   /* orbit velocity magnitude */
    double eVec[3];             /* eccentricity vector */
    double ieHat[3];            /* normalized eccentricity vector */
    double p;                   /* the parameter, also called semi-latus rectum */
    double eps;                 /* small numerical value parameter */

    /* define what is a small numerical value */
    eps = 1e-12;

    /* define inertial frame axes */
    v3Set(1.0, 0.0, 0.0, n1Hat);
    v3Set(0.0, 0.0, 1.0, n3Hat);

    /* norms of position and velocity vectors */
    r = v3Norm(rVec);
    elements->rmag = r;
    v = v3Norm(vVec);
    v3Normalize(rVec, irHat);

    /* Calculate the specific angular momentum and its magnitude */
    v3Cross(rVec, vVec, hVec);
    h = v3Norm(hVec);
    v3Normalize(hVec, ihHat);
	p = h*h / mu;

    /* Calculate the line of nodes */
    v3Cross(n3Hat, hVec, nVec);
    if (v3Norm(nVec) < eps) {
        /* near equatorial orbits */
        v3Copy(n1Hat, inHat);
    } else {
        v3Normalize(nVec, inHat);
    }

    /* Orbit eccentricity vector */
    v3Scale(v * v / mu - 1.0 / r, rVec, eVec);
    v3Scale(v3Dot(rVec, vVec) / mu, vVec, v3);
    v3Subtract(eVec, v3, eVec);
    elements->e = v3Norm(eVec);
	elements->rPeriap = p / (1.0 + elements->e);

    /* Orbit eccentricity unit vector */
    if (elements->e > eps) {
        v3Normalize(eVec, ieHat);
    } else {
        /* near circular orbits, make i_e_hat equal to line of nodes */
        v3Copy(inHat, ieHat);
    }

    /* compute semi-major axis */
    elements->alpha = 2.0 / r - v*v / mu;
    if(fabs(elements->alpha) > eps) {
        /* elliptic or hyperbolic case */
        elements->a = 1.0 / elements->alpha;
        elements->rApoap = p / (1.0 - elements->e);
    } else {
        /* parabolic case */
        elements->a = 0.0;
        elements->rApoap = 0.0;
    }

    /* Calculate the inclination */
    elements->i = safeAcos(hVec[2] / h);

    /* Calculate Ascending Node Omega */
    v3Cross(n1Hat, inHat, v3);
    elements->Omega = atan2(v3[2], inHat[0]);
    if (elements->Omega < 0.0) {
        elements->Omega += 2*M_PI;
    }

    /* Calculate Argument of Periapses omega */
    v3Cross(inHat, ieHat, v3);
    elements->omega = atan2(v3Dot(ihHat,v3), v3Dot(inHat, ieHat));
    if (elements->omega < 0.0) {
        elements->omega += 2*M_PI;
    }

    /* Calculate true anomaly angle f */
    v3Cross(ieHat, irHat, v3);
    elements->f = atan2(v3Dot(ihHat,v3), v3Dot(ieHat, irHat));
    if (elements->f < 0.0) {
        elements->f += 2*M_PI;
    }

    return;
}

/*!
 * Purpose: This program computes the atmospheric density based on altitude
 *   supplied by user.  This function uses a curve fit based on
 *   atmospheric data from the Standard Atmosphere 1976 Data. This
 *   function is valid for altitudes ranging from 100km to 1000km.
 *
 *   Note: This code can only be applied to spacecraft orbiting the Earth
 * Inputs:
 *   alt = altitude in km
 * Outputs:
 *   density = density at the given altitude in kg/m^3
 */
double atmosphericDensity(double alt)
{
    double logdensity;
    double density;
    double val;

    /* Smooth exponential drop-off after 1000 km */
    if(alt > 1000.) {
        logdensity = (-7e-05) * alt - 14.464;
        density = pow(10., logdensity);
        return density;
    }

    /* Calculating the density based on a scaled 6th order polynomial fit to the log of density */
    val = (alt - 526.8000) / 292.8563;
    logdensity = 0.34047 * pow(val, 6) - 0.5889 * pow(val, 5) - 0.5269 * pow(val, 4)
                 + 1.0036 * pow(val, 3) + 0.60713 * pow(val, 2) - 2.3024 * val - 12.575;

    /* Calculating density by raising 10 to the log of density */
    density = pow(10., logdensity);

    return density;
}

/*!
 * Purpose: This program computes the Debye Length length for a given
 *   altitude and is valid for altitudes ranging
 *   from 200 km to GEO (35000km).  However, all values above
 *   1000 km are HIGHLY speculative at this point.
 * Inputs:
 *   alt = altitude in km
 * Outputs:
 *   debye = debye length given in m
 */
double debyeLength(double alt)
{
    double debyedist;
    double a;
    double X[N_DEBYE_PARAMETERS] = {200, 250, 300, 350, 400, 450, 500, 550,
                                    600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,
                                    1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700,
                                    1750, 1800, 1850, 1900, 1950, 2000
                                   };
    double Y[N_DEBYE_PARAMETERS] = {5.64E-03, 3.92E-03, 3.24E-03, 3.59E-03,
                                    4.04E-03, 4.28E-03, 4.54E-03, 5.30E-03, 6.55E-03, 7.30E-03, 8.31E-03,
                                    8.38E-03, 8.45E-03, 9.84E-03, 1.22E-02, 1.37E-02, 1.59E-02, 1.75E-02,
                                    1.95E-02, 2.09E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.47E-02, 2.76E-02,
                                    2.76E-02, 2.76E-02, 2.76E-02, 2.76E-02, 2.76E-02, 2.76E-02, 3.21E-02,
                                    3.96E-02, 3.96E-02, 3.96E-02, 3.96E-02, 3.96E-02
                                   };
    int i;

    /* Flat debyeLength length for altitudes above 2000 km */
    if((alt > 2000.0) && (alt <= 30000.0)) {
        alt = 2000.0;
    } else if((alt > 30000.0) && (alt <= 35000.0)) {
        debyedist = 0.1 * alt - 2999.7;
        return debyedist;
    } else if((alt < 200.0) || (alt > 35000.0)) {
        BSK_PRINT(MSG_ERROR, "debyeLength() received alt = %g\nThe value of alt should be in the range of [200 35000]", alt);
        debyedist = NAN;
        return debyedist;
    }

    /* Interpolation of data */
    for(i = 0; i < N_DEBYE_PARAMETERS - 1; i++) {
        if(X[i + 1] > alt) {
            break;
        }
    }
    a = (alt - X[i]) / (X[i + 1] - X[i]);
    debyedist = Y[i] + a * (Y[i + 1] - Y[i]);

    return debyedist;
}

/*!
 * Purpose: This program computes the atmospheric drag acceleration
 *   vector acting on a spacecraft.
 *   Note the acceleration vector output is inertial, and is
 *   only valid for altitudes up to 1000 km.
 *   Afterwards the drag force is zero. Only valid for Earth.
 * Inputs:
 *   Cd = drag coefficient of the spacecraft
 *   A = cross-sectional area of the spacecraft in m^2
 *   m = mass of the spacecraft in kg
 *   rvec = Inertial position vector of the spacecraft in km  [x;y;z]
 *   vvec = Inertial velocity vector of the spacecraft in km/s [vx;vy;vz]
 * Outputs:
 *   advec = The inertial acceleration vector due to atmospheric
 *             drag in km/sec^2
 */
void atmosphericDrag(double Cd, double A, double m, double *rvec, double *vvec,
                     double *advec)
{
    double r;
    double v;
    double alt;
    double ad;
    double density;

    /* find the altitude and velocity */
    r = v3Norm(rvec);
    v = v3Norm(vvec);
    alt = r - REQ_EARTH;

    /* Checking if user supplied a orbital position is inside the earth */
    if(alt <= 0.) {
        BSK_PRINT(MSG_ERROR, "atmosphericDrag() received rvec = [%g %g %g]The value of rvec should produce a positive altitude for the Earth.",  rvec[1], rvec[2], rvec[3]);
        v3Set(NAN, NAN, NAN, advec);
        return;
    }

    /* get the Atmospheric density at the given altitude in kg/m^3 */
    density = atmosphericDensity(alt);

    /* compute the magnitude of the drag acceleration */
    ad = ((-0.5) * density * (Cd * A / m) * (pow(v * 1000., 2))) / 1000.;

    /* computing the vector for drag acceleration */
    v3Scale(ad / v, vvec, advec);
}

/*!
 * Purpose: Computes the J2_EARTH-J6_EARTH zonal graviational perturbation
 *   accelerations.
 * Inputs:
 *   rvec = Cartesian Position vector in kilometers [x;y;z].
 *   num = Corresponds to which J components to use,
 *       must be an integer between 2 and 6.
 *       (note: Additive- 2 corresponds to J2_EARTH while 3 will
 *       correspond to J2_EARTH + J3_EARTH)
 * Outputs:
 *   ajtot = The total acceleration vector due to the J
 *             perturbations in km/sec^2 [accelx;accely;accelz]
 */
void jPerturb(double *rvec, int num, double *ajtot, ...)
{
    double mu;
    double req;
    double J2, J3, J4, J5, J6;
    double x;
    double y;
    double z;
    double r;
    double temp[3];
    double temp2[3];

    CelestialObject_t planetID;
    va_list args;
    va_start(args, ajtot);
    planetID = (CelestialObject_t) va_arg(args, int);
    va_end(args);

    v3SetZero(ajtot);

    switch(planetID) {
        case CELESTIAL_MERCURY:
            mu  = MU_MERCURY;
            req = REQ_MERCURY;
            J2  = J2_MERCURY;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_VENUS:
            mu  = MU_VENUS;
            req = REQ_VENUS;
            J2  = J2_VENUS;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_MOON:
            mu  = MU_MOON;
            req = REQ_MOON;
            J2  = J2_MOON;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_MARS:
            mu  = MU_MARS;
            req = REQ_MARS;
            J2  = J2_MARS;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_JUPITER:
            mu  = MU_JUPITER;
            req = REQ_JUPITER;
            J2  = J2_JUPITER;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_URANUS:
            mu  = MU_URANUS;
            req = REQ_URANUS;
            J2  = J2_URANUS;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_NEPTUNE:
            mu  = MU_NEPTUNE;
            req = REQ_NEPTUNE;
            J2  = J2_NEPTUNE;
            J3 = 0.0;
            J4 = 0.0;
            J5 = 0.0;
            J6 = 0.0;
            break;

        case CELESTIAL_PLUTO:
        case CELESTIAL_SUN:
            return;

        default:
            mu  = MU_EARTH;
            req = REQ_EARTH;
            J2  = J2_EARTH;
            J3  = J3_EARTH;
            J4  = J4_EARTH;
            J5  = J5_EARTH;
            J6  = J6_EARTH;
            break;
    }


    /* Calculate the J perturbations */
    x = rvec[0];
    y = rvec[1];
    z = rvec[2];
    r = v3Norm(rvec);

    /* Error Checking */
    if((num < 2) || (num > 6)) {
        BSK_PRINT(MSG_ERROR, "jPerturb() received num = %d. The value of num should be 2 <= num <= 6.", num);
        v3Set(NAN, NAN, NAN, ajtot);
        return;
    }

    /* Calculating the total acceleration based on user input */
    if(num >= 2) {
        v3Set((1.0 - 5.0 * pow(z / r, 2.0)) * (x / r),
              (1.0 - 5.0 * pow(z / r, 2.0)) * (y / r),
              (3.0 - 5.0 * pow(z / r, 2.0)) * (z / r), ajtot);
        v3Scale(-3.0 / 2.0 * J2 * (mu / pow(r, 2.0))*pow(req / r, 2.0), ajtot, ajtot);
    }
    if(num >= 3) {
        v3Set(5.0 * (7.0 * pow(z / r, 3.0) - 3.0 * (z / r)) * (x / r),
              5.0 * (7.0 * pow(z / r, 3.0) - 3.0 * (z / r)) * (y / r),
              -3.0 * (10.0 * pow(z / r, 2.0) - (35.0 / 3.0)*pow(z / r, 4.0) - 1.0), temp);
        v3Scale(1.0 / 2.0 * J3 * (mu / pow(r, 2.0))*pow(req / r, 3.0), temp, temp2);
        v3Add(ajtot, temp2, ajtot);
    }
    if(num >= 4) {
        v3Set((3.0 - 42.0 * pow(z / r, 2.0) + 63.0 * pow(z / r, 4.0)) * (x / r),
              (3.0 - 42.0 * pow(z / r, 2.0) + 63.0 * pow(z / r, 4.0)) * (y / r),
              (15.0 - 70.0 * pow(z / r, 2) + 63.0 * pow(z / r, 4.0)) * (z / r), temp);
        v3Scale(5.0 / 8.0 * J4 * (mu / pow(r, 2.0))*pow(req / r, 4.0), temp, temp2);
        v3Add(ajtot, temp2, ajtot);
    }
    if(num >= 5) {
        v3Set(3.0 * (35.0 * (z / r) - 210.0 * pow(z / r, 3.0) + 231.0 * pow(z / r, 5.0)) * (x / r),
              3.0 * (35.0 * (z / r) - 210.0 * pow(z / r, 3.0) + 231.0 * pow(z / r, 5.0)) * (y / r),
              -(15.0 - 315.0 * pow(z / r, 2.0) + 945.0 * pow(z / r, 4.0) - 693.0 * pow(z / r, 6.0)), temp);
        v3Scale(1.0 / 8.0 * J5 * (mu / pow(r, 2.0))*pow(req / r, 5.0), temp, temp2);
        v3Add(ajtot, temp2, ajtot);
    }
    if(num >= 6) {
        v3Set((35.0 - 945.0 * pow(z / r, 2) + 3465.0 * pow(z / r, 4.0) - 3003.0 * pow(z / r, 6.0)) * (x / r),
              (35.0 - 945.0 * pow(z / r, 2.0) + 3465.0 * pow(z / r, 4.0) - 3003.0 * pow(z / r, 6.0)) * (y / r),
              -(3003.0 * pow(z / r, 6.0) - 4851.0 * pow(z / r, 4.0) + 2205.0 * pow(z / r, 2.0) - 245.0) * (z / r), temp);
        v3Scale(-1.0 / 16.0 * J6 * (mu / pow(r, 2.0))*pow(req / r, 6.0), temp, temp2);
        v3Add(ajtot, temp2, ajtot);
    }
}

/*!
 * Purpose: Computes the inertial solar radiation force vectors
 *   based on cross-sectional Area and mass of the spacecraft
 *   and the position vector of the planet to the sun.
 *   Note: It is assumed that the solar radiation pressure decreases
 *   quadratically with distance from sun (in AU)
 *
 *   Solar Radiation Equations obtained from
 *   Earth Space and Planets Journal Vol. 51, 1999 pp. 979-986
 * Inputs:
 *   A = Cross-sectional area of the spacecraft that is facing
 *             the sun in m^2.
 *   m = The mass of the spacecraft in kg.
 *   sunvec = Position vector to the Sun in units of AU.
 *             Earth has a distance of 1 AU.
 * Outputs:
 *   arvec = The inertial acceleration vector due to the effects
 *       of Solar Radiation pressure in km/sec^2.  The vector
 *       components of the output are the same as the vector
 *       components of the sunvec input vector.
 */
void solarRad(double A, double m, double *sunvec, double *arvec)
{
    double flux;
    double c;
    double Cr;
    double sundist;

    /* Solar Radiation Flux */
    flux = 1372.5398;           /* Watts/m^2 */

    /* Speed of light */
    c = 299792458.;             /* m/s */

    /* Radiation pressure coefficient */
    Cr = 1.3;

    /* Magnitude of position vector */
    sundist = v3Norm(sunvec);   /* AU */

    /* Computing the acceleration vector */
    v3Scale((-Cr * A * flux) / (m * c * pow(sundist, 3)) / 1000., sunvec, arvec);
}

/*! maps classical mean orbit elements to Osculating elements */
void clMeanOscMap(double req, double J2, classicElements *elements, classicElements *elements_p, double sgn) {
    // Classical orbital elements = (a,e,i,Omega,omega,f)
    // First-order J2 Mapping Between Mean and Osculating Orbital Elements
    // sgn=1:mean2osc, sgn=-1:osc2mean
    // Analytical Mechanics of Space Systems
    // Hanspeter Schaub, John L. Junkins, 4th edition.
    // [m] or [km] should be the same both for req and elements->a

    double a     = elements->a;
    double e     = elements->e;
    double i     = elements->i;
    double Omega = elements->Omega;
    double omega = elements->omega;
    double f     = elements->f;
    double E     = f2E(f, e);
    double M     = E2M(E, e);

    double gamma2 = sgn * J2 / 2.0 * pow((req / a), 2.0);  // (F.1),(F.2)
    double eta = sqrt(1.0 - pow(e, 2.0));
    double gamma2p = gamma2 / pow(eta, 4.0);        // (F.3)
    double a_r = (1.0 + e * cos(f)) / pow(eta, 2.0);  // (F.6)

    double ap = a + a * gamma2 * ((3.0 * pow(cos(i), 2.0) - 1.0) * (pow(a_r, 3.0) - 1.0 / pow(eta, 3.0))
              + 3.0 * (1.0 - pow(cos(i), 2.0)) * pow(a_r, 3.0) * cos(2.0 * omega + 2.0 * f));  // (F.7)

    double de1 = gamma2p / 8.0 * e * pow(eta, 2.0) * (1.0 - 11.0 * pow(cos(i), 2.0) - 40.0 * pow(cos(i), 4.0)
               / (1.0 - 5.0 * pow(cos(i), 2.0))) * cos(2.0 * omega);  // (F.8)

    double de = de1 + pow(eta, 2.0) / 2.0 * (gamma2 * ((3.0 * pow(cos(i), 2.0) - 1.0) / pow(eta, 6.0) * (e * eta + e / (1.0 + eta) + 3.0 * cos(f)
              + 3.0 * e * pow(cos(f), 2.0) + pow(e, 2.0) * pow(cos(f), 3.0)) + 3.0 * (1.0 - pow(cos(i), 2.0)) / pow(eta, 6.0) * (e + 3.0 * cos(f)
              + 3 * e * pow(cos(f), 2) + pow(e, 2) * pow(cos(f), 3)) * cos(2 * omega + 2 * f))
              - gamma2p * (1.0 - pow(cos(i), 2.0)) * (3.0 * cos(2.0 * omega + f) + cos(2.0 * omega + 3.0 * f)));  // (F.9)

    double di = -e * de1 / pow(eta, 2.0) / tan(i)
              + gamma2p / 2.0 * cos(i) * sqrt(1.0 - pow(cos(i), 2.0)) * (3.0 * cos(2.0 * omega + 2.0 * f) + 3.0 * e * cos(2.0 * omega + f)
              + e * cos(2.0 * omega + 3.0 * f));  // (F.10)

    double MpopOp = M + omega + Omega + gamma2p / 8.0 * pow(eta, 3.0) * (1.0 - 11.0 * pow(cos(i), 2.0) - 40.0 * pow(cos(i), 4.0)
                  / (1.0 - 5.0 * pow(cos(i), 2.0))) * sin(2.0 * omega) - gamma2p / 16.0 * (2.0 + pow(e, 2.0) - 11.0 * (2.0 + 3.0 * pow(e, 2.0))
                  * pow(cos(i), 2.0) - 40.0 * (2.0 + 5.0 * pow(e, 2.0)) * pow(cos(i), 4.0) / (1.0 - 5.0 * pow(cos(i), 2.0)) - 400.0 * pow(e, 2.0)
                  * pow(cos(i), 6.0) / pow(1.0 - 5.0 * pow(cos(i), 2.0), 2.0)) * sin(2.0 * omega) + gamma2p / 4.0 * (-6.0 * (1.0 - 5.0 * pow(cos(i), 2.0))
                  * (f - M + e * sin(f)) + (3.0 - 5.0 * pow(cos(i), 2.0)) * (3.0 * sin(2.0 * omega + 2.0 * f) + 3.0 * e * sin(2.0 * omega + f) + e
                  * sin(2.0 * omega + 3.0 * f))) - gamma2p / 8.0 * pow(e, 2.0) * cos(i) * (11.0 + 80.0 * pow(cos(i), 2.0) / (1.0 - 5.0 * pow(cos(i), 2.0))
                  + 200.0 * pow(cos(i), 4.0) / pow(1.0 - 5.0 * pow(cos(i), 2.0), 2.0)) * sin(2.0 * omega) - gamma2p / 2.0 * cos(i)
                  * (6.0 * (f - M + e * sin(f)) - 3.0 * sin(2.0 * omega + 2.0 * f) - 3.0 * e * sin(2.0 * omega + f) - e * sin(2.0 * omega + 3.0 * f));  // (F.11)

    double edM = gamma2p / 8.0 * e * pow(eta, 3.0) * (1.0 - 11.0 * pow(cos(i), 2.0)
               - 40.0 * pow(cos(i), 4.0) / (1.0 - 5.0 * pow(cos(i), 2.0))) * sin(2.0 * omega)
               - gamma2p / 4.0 * pow(eta, 3.0) * (2.0 * (3.0 * pow(cos(i), 2.0) - 1.0) * (pow(a_r * eta, 2.0) + a_r + 1.0) * sin(f)
               + 3.0 * (1.0 - pow(cos(i), 2.0)) * ((-pow(a_r * eta, 2.0) - a_r + 1.0) * sin(2.0 * omega + f) + (pow(a_r * eta, 2.0) + a_r
               + 1.0 / 3.0) * sin(2.0 * omega + 3.0 * f)));  // (F.12)

    double dOmega = -gamma2p / 8.0 * pow(e, 2.0) * cos(i) * (11.0 + 80.0 * pow(cos(i), 2.0) / (1.0 - 5.0 * pow(cos(i), 2.0))
                  + 200.0 * pow(cos(i), 4.0) / pow(1.0 - 5.0 * pow(cos(i), 2.0), 2.0)) * sin(2.0 * omega) - gamma2p / 2.0 * cos(i)
                  * (6.0 * (f - M + e * sin(f)) - 3.0 * sin(2.0 * omega + 2.0 * f) - 3.0 * e * sin(2.0 * omega + f) - e * sin(2.0 * omega + 3.0 * f));  // (F.13)

    double d1 = (e + de) * sin(M) + edM * cos(M);  // (F.14)
    double d2 = (e + de) * cos(M) - edM * sin(M);  // (F.15)

    double Mp = atan2(d1, d2);            // (F.16)
    double ep = sqrt(d1 * d1 + d2 * d2);  // (F.17)

    double d3 = (sin(i / 2.0) + cos(i / 2.0) * di / 2.0) * sin(Omega) + sin(i / 2.0) * dOmega * cos(Omega);  // (F.18)
    double d4 = (sin(i / 2.0) + cos(i / 2.0) * di / 2.0) * cos(Omega) - sin(i / 2.0) * dOmega * sin(Omega);  // (F.19)

    double Omegap = atan2(d3, d4);                  // (F.20)
    double ip = 2.0 * safeAsin(sqrt(d3 * d3 + d4 * d4));  // (F.21)
    double omegap = MpopOp - Mp - Omegap;           // (F.22)

    double Ep = M2E(Mp, ep);
    double fp = E2f(Ep, ep);

    elements_p->a = ap;
    elements_p->e = ep;
    elements_p->i = ip;
    elements_p->Omega = Omegap;
    elements_p->omega = omegap;
    elements_p->f = fp;
    return;
}

/*! maps from classical orbit elements to equinoctial elements */
void clElem2eqElem(classicElements *elements_cl, equinoctialElements *elements_eq) {
    // conversion
    // from classical orbital elements (a,e,i,Omega,omega,f)
    // to equinoctial orbital elements (a,P1,P2,Q1,Q2,l,L)
    elements_eq->a  = elements_cl->a;
    elements_eq->P1 = elements_cl->e * sin(elements_cl->Omega + elements_cl->omega);
    elements_eq->P2 = elements_cl->e * cos(elements_cl->Omega + elements_cl->omega);
    elements_eq->Q1 = tan(elements_cl->i / 2.0) * sin(elements_cl->Omega);
    elements_eq->Q2 = tan(elements_cl->i / 2.0) * cos(elements_cl->Omega);
    double E        = f2E(elements_cl->f, elements_cl->e);
    double M        = E2M(E, elements_cl->e);
    elements_eq->l  = elements_cl->Omega + elements_cl->omega + M;
    elements_eq->L  = elements_cl->Omega + elements_cl->omega + elements_cl->f;
    return;
}
