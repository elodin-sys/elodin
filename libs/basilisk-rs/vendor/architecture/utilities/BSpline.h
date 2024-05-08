/*
 ISC License

 Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder

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


#include <Eigen/Dense>
#include "architecture/utilities/avsEigenSupport.h"
#include "architecture/utilities/macroDefinitions.h"


//! @brief The InputDataSet class contains the information about the points that must be interpolated.
//! It is used as a data structure to intialize the inputs that are passed to the interpolating function.

class InputDataSet {
public:
    InputDataSet();
    InputDataSet(Eigen::VectorXd X1, Eigen::VectorXd X2, Eigen::VectorXd X3);
    ~InputDataSet();

    void setXDot_0(Eigen::Vector3d XDot_0);
    void setXDot_N(Eigen::Vector3d XDot_N);
    void setXDDot_0(Eigen::Vector3d XDDot_0);
    void setXDDot_N(Eigen::Vector3d XDDot_N);
    void setT(Eigen::VectorXd T);
    void setW(Eigen::VectorXd W);
    void setAvgXDot(double AvgXDot);
    
    double AvgXDot;                  //!< desired average velocity norm
    Eigen::VectorXd T;               //!< time tags: specifies at what time each waypoint is hit
    Eigen::VectorXd W;               //!< weight vector for the LS approximation
    Eigen::VectorXd X1;              //!< coordinate #1 of the waypoints
    Eigen::VectorXd X2;              //!< coordinate #2 of the waypoints
    Eigen::VectorXd X3;              //!< coordinate #3 of the waypoints
    Eigen::Vector3d XDot_0;          //!< 3D vector containing the first derivative at starting point
    Eigen::Vector3d XDot_N;          //!< 3D vector containing the first derivative at final point
    Eigen::Vector3d XDDot_0;         //!< 3D vector containing the second derivative at starting point
    Eigen::Vector3d XDDot_N;         //!< 3D vector containing the second derivative at final point
    bool T_flag;                     //!< indicates that time tags have been specified; if true, AvgXDot_flag is false
    bool AvgXDot_flag;               //!< indicates that avg velocity norm has been specified; if true, T_flag is false
    bool W_flag;                     //!< indicates that weight vector has been specified
    bool XDot_0_flag;                //!< indicates that first derivative at starting point has been specified
    bool XDot_N_flag;                //!< indicates that first derivative at final point has been specified
    bool XDDot_0_flag;               //!< indicates that second derivative at starting point has been specified
    bool XDDot_N_flag;               //!< indicates that second derivative at final point has been specified
};

//! @brief The OutputDataSet class is used as a data structure to contain the interpolated function and its first- and 
//! second-order derivatives, together with the time-tag vector T.
class OutputDataSet {
public:
    OutputDataSet();
    ~OutputDataSet();
    void getData(double t, double x[3], double xDot[3], double xDDot[3]);
    double getStates(double t, int derivative,  int index);
    
    Eigen::VectorXd T;               //!< time tags for each point of the interpolated trajectory
    Eigen::VectorXd X1;              //!< coordinate #1 of the interpolated trajectory
    Eigen::VectorXd X2;              //!< coordinate #2 of the interpolated trajectory
    Eigen::VectorXd X3;              //!< coordinate #3 of the interpolated trajectory
    Eigen::VectorXd XD1;             //!< first derivative of coordinate #1 of the interpolated trajectory
    Eigen::VectorXd XD2;             //!< first derivative of coordinate #2 of the interpolated trajectory
    Eigen::VectorXd XD3;             //!< first derivative of coordinate #3 of the interpolated trajectory
    Eigen::VectorXd XDD1;            //!< second derivative of coordinate #1 of the interpolated trajectory
    Eigen::VectorXd XDD2;            //!< second derivative of coordinate #2 of the interpolated trajectory
    Eigen::VectorXd XDD3;            //!< second derivative of coordinate #3 of the interpolated trajectory

    int P;                           //!< polynomial degree of the BSpline
    Eigen::VectorXd U;               //!< knot vector of the BSpline
    Eigen::VectorXd C1;              //!< coordinate #1 of the control points
    Eigen::VectorXd C2;              //!< coordinate #2 of the control points
    Eigen::VectorXd C3;              //!< coordinate #3 of the control points
};

void interpolate(InputDataSet Input, int Num, int P, OutputDataSet *Output);

void approximate(InputDataSet Input, int Num, int Q, int P, OutputDataSet *Output);

void basisFunction(double t, Eigen::VectorXd U, int I, int P, double *NN, double *NN1, double *NN2);