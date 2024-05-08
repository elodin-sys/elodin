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

#pragma once

#include <Eigen/Dense>
#include <architecture/utilities/orbitalMotion.h>
#include "architecture/utilities/astroConstants.h"


//! @brief The KeplerianOrbit class represents an elliptical orbit and provides a coherent set of
//! common outputs such as position and velocity, orbital period, semi-parameter, etc. It uses the
//! utility orbitalMotion to do orbital element to position and velocity conversion.
class KeplerianOrbit {
public:
    KeplerianOrbit();
    KeplerianOrbit(classicElements oe, const double mu);
    KeplerianOrbit(const KeplerianOrbit &orig);
    ~KeplerianOrbit();


    Eigen::Vector3d r_BP_P() const;              //!< body position vector relative to planet
    Eigen::Vector3d v_BP_P() const;              //!< body velocity vector relative to planet
    Eigen::Vector3d h_BP_P() const; //!< angular momentum of body relative to planet
    double M() const;
    double n() const;
    double P() const;
    double f() const;
    double fDot() const;
    double RAAN() const;
    double omega() const;
    double i() const;
    double e() const;
    double a() const;
    double h() const;
    double Energy();
    double r() const;
    double v() const;
    double r_a() const;
    double r_p() const;
    double fpa() const;
    double E() const;
    double p() const;
    double rDot() const;
    double c3() const;
    classicElements oe();
    void set_mu(const double mu);
    void set_a(double a);
    void set_e(double e);
    void set_i(double i);
    void set_omega(double omega);
    void set_RAAN(double RAAN);
    void set_f(double f);
    
private:
    double mu = MU_EARTH;
    double semi_major_axis = 1E5;
    double eccentricity = 1E-5;
    double inclination{};
    double argument_of_periapsis{};
    double right_ascension{};
    double true_anomaly{};
    double true_anomaly_rate{};
    double orbital_period{};
    double orbital_energy{};
    double v_infinity{};
    double orbit_radius{};
    double radial_rate{};
    double r_apogee{};
    double r_perigee{};
    double semi_parameter{};
    double flight_path_angle{};
    double eccentric_anomaly{};
    double mean_motion{};
    double mean_anomaly{};
    Eigen::Vector3d orbital_angular_momentum_P;
    Eigen::Vector3d position_BP_P;
    Eigen::Vector3d velocity_BP_P;
private:
    void change_orbit();
    void change_f();
};

