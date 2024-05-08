/*
ISC License

Copyright (c) 2016-2017, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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

#ifndef _GEODETIC_CONV_H_
#define _GEODETIC_CONV_H_

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <Eigen/Dense>


/*! @brief Collection of utility functions for converting in/out of planet-centric reference frames.

The geodeticConversion library contains simple transformations between inertial coordinates and planet-fixed coordinates in a general way.

No support is provided for non-spherical bodies. Transformations are scripted from Vallado.

 */

Eigen::Vector3d PCI2PCPF(Eigen::Vector3d pciPosition, double J20002Pfix[3][3]);
Eigen::Vector3d PCPF2LLA(Eigen::Vector3d pciPosition, double planetEqRadius, double planetPoRad=-1.0);
Eigen::Vector3d PCI2LLA(Eigen::Vector3d pciPosition, double J20002Pfix[3][3], double planetEqRad, double planetPoRad=-1.0);
Eigen::Vector3d LLA2PCPF(Eigen::Vector3d llaPosition, double planetEqRad, double planetPoRad=-1.0);
Eigen::Vector3d PCPF2PCI(Eigen::Vector3d pcpfPosition, double J20002Pfix[3][3]);
Eigen::Vector3d LLA2PCI(Eigen::Vector3d llaPosition, double J20002Pfix[3][3], double planetEqRad, double planetPoRad=-1.0);
Eigen::Matrix3d C_PCPF2SEZ(double lat, double longitude);

#endif
