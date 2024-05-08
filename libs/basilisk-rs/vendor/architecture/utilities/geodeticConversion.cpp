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
#include "geodeticConversion.h"
#include "rigidBodyKinematics.h"
#include "avsEigenSupport.h"


/*! Converts from a planet-centered inertial position (i.e., J2000 ECI) to a planet-centered, planet-fixed position given a rotation matrix.
@param pciPosition: [m] Position vector in PCI coordinates
@param J20002Pfix: [-] 3x3 rotation matrix representing the rotation between PCPF and ECI frames
@return pcpfPosition: [m] Position vector in PCPF coordinates
*/
Eigen::Vector3d PCI2PCPF(Eigen::Vector3d pciPosition, double J20002Pfix[3][3]){
    // cArray2EigenMatrixd expects a column major input, thus the result is transposed
    Eigen::MatrixXd dcm_PN = cArray2EigenMatrixXd(*J20002Pfix,3,3).transpose();
    Eigen::Vector3d pcpfPosition = dcm_PN * pciPosition;

    return pcpfPosition;
}

/*! Converts from a planet-centered, planet-fixed coordinates to latitude/longitude/altitude (LLA) coordinates given a planet radius.
@param pcpfPosition: [m] Position vector in PCPF coordinates
@param planetEqRad: [m] Planetary radius, assumed to be constant (i.e., spherical)
@param planetPoRad: [m] Planetary polar used for elliptical surfaces if provided, otherwise spherical,
@return llaPosition: Final position in latitude/longitude/altitude coordinates
  [0] : [rad] latitude above planetary equator
  [1] : [rad] longitude across planetary meridian
  [2] : [alt] altitude above planet radius
*/
Eigen::Vector3d PCPF2LLA(Eigen::Vector3d pcpfPosition, double planetEqRad, double planetPoRad){

  Eigen::Vector3d llaPosition;
  llaPosition.fill(0.0);
  if(planetPoRad < 0.0) {
      llaPosition[2] =
              pcpfPosition.norm() - planetEqRad; //  Altitude is the height above assumed-spherical planet surface
      llaPosition[1] = atan2(pcpfPosition[1], pcpfPosition[0]);
      llaPosition[0] = atan2(pcpfPosition[2], sqrt(pow(pcpfPosition[0], 2.0) + pow(pcpfPosition[1], 2.0)));
  }
  else
  {

      const int nr_kapp_iter = 10;
      const double good_kappa_acc = 1.0E-13;
      double planetEcc2 = 1.0 - planetPoRad*planetPoRad/(planetEqRad*planetEqRad);
      double kappa = 1.0/(1.0-planetEcc2);
      double p2 = pcpfPosition[0]*pcpfPosition[0] + pcpfPosition[1]*pcpfPosition[1];
      double z2 = pcpfPosition[2]*pcpfPosition[2];
      for(int i=0; i<nr_kapp_iter; i++)
      {
          double cI = (p2 + (1.0-planetEcc2)*z2*kappa*kappa);
          cI = sqrt(cI*cI*cI)/(planetEqRad*planetEcc2);
          double kappaNext = 1.0 + (p2 + (1.0-planetEcc2)*z2*kappa*kappa*kappa)/(cI-p2);
          if(std::abs(kappaNext-kappa) < good_kappa_acc)
          {
              break;
          }
          kappa = kappaNext;
      }
      double pSingle = sqrt(p2);
      llaPosition[0] = atan2(kappa*pcpfPosition[2], pSingle);
      double sPhi = sin(llaPosition[0]);
      double nVal = planetEqRad/sqrt(1.0 - planetEcc2*sPhi*sPhi);
      llaPosition[2] = pSingle/(cos(llaPosition[0])) - nVal;
      llaPosition[1] = atan2(pcpfPosition[1], pcpfPosition[0]);
  }

  return llaPosition;
}

/*! Converts from a planet-centered inertial coordinates to latitutde/longitude/altitude (LLA) coordinates given a planet radius and rotation matrix.
@param pciPosition: [m] Position vector in PCPF coordinates
@param J20002Pfix: planet DCM
@param planetEqRad: [m] Planetary radius, assumed to be constant (i.e., spherical)
@param planetPoRad: [m] Planetary polar used for elliptical surfaces if provided, otherwise spherical,
@return llaPosition: Final position in latitude/longitude/altitude coordinates
  [0] : [rad] latitude above planetary equator
  [1] : [rad] longitude across planetary meridian
  [2] : [alt] altitude above planet radius
*/
Eigen::Vector3d PCI2LLA(Eigen::Vector3d pciPosition, double J20002Pfix[3][3], double planetEqRad, double planetPoRad){
  Eigen::Vector3d pcpfVec = PCI2PCPF(pciPosition, J20002Pfix);
  Eigen::Vector3d llaVec = PCPF2LLA(pcpfVec, planetEqRad, planetPoRad);

  return llaVec;
}

/*! Converts from a Lat/Long/Altitude coordinates to planet-centered,planet-fixed coordinates given a planet radius.
@param llaPosition: [m] Position vector in PCPF coordinates
@param planetEqRad: [m] Planetary equatorial radius, assumed to be constant (i.e., spherical)
@param planetPoRad: [m] Planetary polar used for elliptical surfaces if provided, otherwise spherical,
@return pcpfPosition: [m] Final position in the planet-centered, planet-fixed frame.
*/
Eigen::Vector3d LLA2PCPF(Eigen::Vector3d llaPosition, double planetEqRad, double planetPoRad){
    Eigen::Vector3d pcpfPosition;
    pcpfPosition.fill(0.0);
    double planetEcc2 = 0.0;
    if(planetPoRad >= 0)
    {
        planetEcc2 = 1.0 - planetPoRad*planetPoRad/(planetEqRad*planetEqRad);
    }
    double sPhi = sin(llaPosition[0]);
    double nVal = planetEqRad/sqrt(1.0 - planetEcc2*sPhi*sPhi);
    pcpfPosition[0] = (nVal+llaPosition[2])*cos(llaPosition[0])*cos(llaPosition[1]);
    pcpfPosition[1] = (nVal+llaPosition[2])*cos(llaPosition[0])*sin(llaPosition[1]);
    pcpfPosition[2] = ((1.0-planetEcc2)*nVal + llaPosition[2])*sPhi;

    return pcpfPosition;
}

/*! Converts from a Lat/Long/Altitude coordinates to planet-centered inertialcoordinates given a planet radius and rotation matrix.
@param pcpfPosition: [m] Position vector in planet centered, planet fixed coordinates
@param J20002Pfix: [-] Rotation between inertial and pcf coordinates.
@return pciPosition: [m] Final position in the planet-centered inertial frame.
*/
Eigen::Vector3d PCPF2PCI(Eigen::Vector3d pcpfPosition, double J20002Pfix[3][3])
{
    // cArray2EigenMatrixd expects a column major input, thus the result is transposed
    Eigen::MatrixXd dcm_NP = cArray2EigenMatrixXd(*J20002Pfix,3,3);
    Eigen::Vector3d pciPosition = dcm_NP * pcpfPosition;

    return pciPosition;
}

/*! Converts from a planet-centered inertial coordinates to latitutde/longitude/altitude (LLA) coordinates given a planet radius and rotation matrix.
@param llaPosition: Final position in latitude/longitude/altitude coordinates
  [0] : [rad] latitude above planetary equator
  [1] : [rad] longitude across planetary meridian
  [2] : [alt] altitude above planet radius
@param J20002Pfix: rotation matrix between inertial and PCPF frames
@param planetEqRad: [m] Planetary radius, assumed to be constant (i.e., spherical)
@param planetPoRad: [m] Planetary polar used for elliptical surfaces if provided, otherwise spherical,
@return pciPosition : [m] Position in inertial coordinates.
*/
Eigen::Vector3d LLA2PCI(Eigen::Vector3d llaPosition, double J20002Pfix[3][3], double planetEqRad, double planetPoRad)
{
  Eigen::Vector3d pcpfPosition = LLA2PCPF(llaPosition, planetEqRad, planetPoRad);
  Eigen::Vector3d pciPosition = PCPF2PCI(pcpfPosition, J20002Pfix);
  return pciPosition;
}

Eigen::Matrix3d C_PCPF2SEZ(double lat, double longitude)
{
     double m1[3][3];
     double m2[3][3];
     Euler2(M_PI_2-lat, m1);
     Euler3(longitude, m2);

     Eigen::MatrixXd rot2 = cArray2EigenMatrix3d(*m1);
     Eigen::MatrixXd rot3 = cArray2EigenMatrix3d(*m2);
     return rot2*rot3;
}
