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

#include "BSpline.h"
#include <architecture/utilities/avsEigenSupport.h>
#include <iostream>
#include <cstring>
#include <math.h>

/*! This constructor initializes an Input structure for BSpline interpolation */
InputDataSet::InputDataSet()
{
    return;
}

/*! The constructor requires 3 N-dimensional vectors containing the coordinates of the waypoints */
InputDataSet::InputDataSet(Eigen::VectorXd X1, Eigen::VectorXd X2, Eigen::VectorXd X3)
{
    this->X1 = X1;
    this->X2 = X2;
    this->X3 = X3;
    this->XDot_0_flag = false;
    this->XDot_N_flag = false;
    this->XDDot_0_flag = false;
    this->XDDot_N_flag = false;
    this->T_flag = false;
    this->AvgXDot_flag = false;
    this->W_flag = false;

    uint64_t N1 = (uint64_t) X1.size();
    uint64_t N2 = (uint64_t) X2.size();
    uint64_t N3 = (uint64_t) X3.size();

    if ((N1 != N2) || (N1 != N3) || (N2 != N3)) {
        std::cout << "Error in BSpline.InputDataSet: \n the Input coordinate vectors X1, X2, X3 have different sizes. \n";
    }

    return;
}

/*! Generic destructor */
InputDataSet::~InputDataSet()
{
    return;
}

/*! Set the first derivative of the starting point (optional) */
void InputDataSet::setXDot_0(Eigen::Vector3d XDot_0) {this->XDot_0 = XDot_0; this->XDot_0_flag = true; return;}

/*! Set the first derivative of the last point (optional) */
void InputDataSet::setXDot_N(Eigen::Vector3d XDot_N) {this->XDot_N = XDot_N; this->XDot_N_flag = true; return;}

/*! Set the second derivative of the starting point (optional) */
void InputDataSet::setXDDot_0(Eigen::Vector3d XDDot_0) {this->XDDot_0 = XDDot_0; this->XDDot_0_flag = true; return;}

/*! Set the second derivative of the last point (optional) */
void InputDataSet::setXDDot_N(Eigen::Vector3d XDDot_N) {this->XDDot_N = XDDot_N; this->XDDot_N_flag = true; return;}

/*! Set the time tags for each waypoint (optional). Cannot be imposed together with avg velocity norm below */
void InputDataSet::setT(Eigen::VectorXd T) {this->T = T; this->T_flag = true; this->AvgXDot_flag = false; return;}

/*! Set the average velocity norm (optional). Cannot be imposed together with time tag vector above */
void InputDataSet::setAvgXDot(double AvgXDot) {this->AvgXDot = AvgXDot; this->AvgXDot_flag = true; this->T_flag = false; return;}

/*! Set the weights for each waypoint. Weights are used in the LS approximation */
void InputDataSet::setW(Eigen::VectorXd W) {this->W = W; this->W_flag = true; return;}

/*! This constructor initializes an Output structure for BSpline interpolation */
OutputDataSet::OutputDataSet()
{
    return;
}

/*! Generic destructor */
OutputDataSet::~OutputDataSet()
{
    return;
}

/*! This method returns x, xDot and xDDot at the desired input time T */
void OutputDataSet::getData(double T, double x[3], double xDot[3], double xDDot[3])
{
    int N = (int) this->T.size() - 1;
    double Ttot = this->T[N];

    // if T < Ttot calculalte the values
    if (T <= Ttot) {
        double t = T / Ttot;
        int Q = (int) this->C1.size();
        Eigen::VectorXd NN(Q), NN1(Q), NN2(Q);
        basisFunction(t, this->U, Q, this->P, &NN[0], &NN1[0], &NN2[0]);
        x[0] = NN.dot(this->C1);
        x[1] = NN.dot(this->C2);
        x[2] = NN.dot(this->C3);
        xDot[0] = NN1.dot(this->C1) / Ttot;
        xDot[1] = NN1.dot(this->C2) / Ttot;
        xDot[2] = NN1.dot(this->C3) / Ttot;
        xDDot[0] = NN2.dot(this->C1) / pow(Ttot,2);
        xDDot[1] = NN2.dot(this->C2) / pow(Ttot,2);
        xDDot[2] = NN2.dot(this->C3) / pow(Ttot,2);
    }
    // if t > Ttot return final value with zero derivatives
    else {
        x[0] = this->X1[N];
        x[1] = this->X2[N];
        x[2] = this->X3[N];
        xDot[0] = 0;
        xDot[1] = 0;
        xDot[2] = 0;
        xDDot[0] = 0;
        xDDot[1] = 0;
        xDDot[2] = 0;
    }
}

/*! This method returns single coordinates of x, xDot and xDDot at the desired input time T. */
/*! It is designed to be accessible from Python */
double OutputDataSet::getStates(double T, int derivative, int index)
{
    int N = (int) this->T.size()-1;
    double Ttot = this->T[N];

    // if T < Ttot calculalte the values
    if (T <= Ttot) {
        double t = T / Ttot;
        int Q = (int) this->C1.size();
        Eigen::VectorXd NN(Q), NN1(Q), NN2(Q);
        basisFunction(t, this->U, Q, this->P, &NN[0], &NN1[0], &NN2[0]);

        switch (derivative) {
            case 0 :
                switch(index) {
                    case 0 :
                        return NN.dot(this->C1);
                    case 1 :
                        return NN.dot(this->C2);
                    case 2 :
                        return NN.dot(this->C3);
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            case 1 :
                switch(index) {
                    case 0 :
                        return NN1.dot(this->C1) / Ttot;
                    case 1 :
                        return NN1.dot(this->C2) / Ttot;
                    case 2 :
                        return NN1.dot(this->C3) / Ttot;
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            case 2 :
                switch(index) {
                    case 0 :
                        return NN2.dot(this->C1) / pow(Ttot,2);
                    case 1 :
                        return NN2.dot(this->C2) / pow(Ttot,2);
                    case 2 :
                        return NN2.dot(this->C3) / pow(Ttot,2);
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            default :
                std::cout << "Error in Output.getStates: invalid derivative \n";
                return 1000;
        }
    }
    // if t > Ttot return final value with zero derivatives
    else {
        switch (derivative) {
            case 0 :
                switch(index) {
                    case 0 :
                        return this->X1[N];
                    case 1 :
                        return this->X2[N];
                    case 2 :
                        return this->X3[N];
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            case 1 :
                switch(index) {
                    case 0 :
                        return 0;
                    case 1 :
                        return 0;
                    case 2 :
                        return 0;
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            case 2 :
                switch(index) {
                    case 0 :
                        return 0;
                    case 1 :
                        return 0;
                    case 2 :
                        return 0;
                    default :
                        std::cout << "Error in Output.getStates: invalid index \n";
                        return 1000;
                }
            default :
                std::cout << "Error in Output.getStates: invalid derivative \n";
                return 1000;
        }
    }
}

/*! This function takes the Input structure, performs the BSpline interpolation and outputs the result into Output structure */
void interpolate(InputDataSet Input, int Num, int P, OutputDataSet *Output)
{   
    Output->P = P;

    // N = number of waypoints - 1 
    int N = (int) Input.X1.size() - 1;
    
    // T = time tags; if not specified, it is computed from a cartesian distance assuming a constant velocity norm on average
    Eigen::VectorXd T(N+1);
    double S = 0;
    if (Input.T_flag == true) {
        T = Input.T;
    }
    else {
        T[0] = 0;
        for (int n = 1; n < N+1; n++) {
            T[n] = T[n-1] + pow( (pow(Input.X1[n]-Input.X1[n-1], 2) + pow(Input.X2[n]-Input.X2[n-1], 2) + pow(Input.X3[n]-Input.X3[n-1], 2)), 0.5 );
            S += T[n] - T[n-1];
        }
    }
    if (Input.AvgXDot_flag == true) {
        for (int n = 0; n < N+1; n++) {
            T[n] = T[n] / T[N] * S / Input.AvgXDot;
        }
    }

    double Ttot = T[N];

    // build uk vector: normalized waypoint time tags
    Eigen::VectorXd uk(N+1);
    for (int n = 0; n < N+1; n++) {
        uk[n] = T[n] / Ttot;
    }

    // K = number of endpoint derivatives
    int K = 0;
    if (Input.XDot_0_flag == true) {K += 1;}
    if (Input.XDot_N_flag == true) {K += 1;}
    if (Input.XDDot_0_flag == true) {K += 1;}
    if (Input.XDDot_N_flag == true) {K += 1;}
    
    // The maximum polynomial order is N + K. If a higher order is requested, print a BSK_ERROR
    if (P > N + K) {
        std::cout << "Error in BSpline.interpolate: \n the desired polynomial order P is too high. Mass matrix A will be singular. \n" ;
    }

    int M = N + P + K + 1;

    // build knot vector U of size M + 1
    Eigen::VectorXd U(M+1);
    double u;
    for (int p = 0; p < P+1; p++) {
        U[p] = 0;
    }
    for (int j = 0; j < M-2*P-1; j++) {
        u = 0.0;
        for (int i = j; i < j+P; i++) {
            if (i >= uk.size()) {
                u += uk[N] / P;
            }
            else {
                u += uk[i] / P;
            }
        U[P+j+1] = u;
        }
    }
    for (int p = 0; p < P+1; p++) {
        U[M-P+p] = 1;
    }

    // build stiffness matrix A of size (N+K+1)x(N+K+1)
    Eigen::MatrixXd A(N+K+1,N+K+1);
    // build vectors Q1, Q2, Q3 (left hand side of linear system)
    Eigen::VectorXd Q1(N+K+1), Q2(N+K+1), Q3(N+K+1);
    // populate A with zeros
    for (int n = 0; n < N+K+1; n++) {
        for (int m = 0; m < N+K+1; m++) {
            A(n,m) = 0;
        }
    }
    int n = -1;
    // constrain first derivative at starting point
    if (Input.XDot_0_flag == true) {
        n += 1;
        A(n,0) = -1;
        A(n,1) =  1;
        Q1[n] = U[P+1] / P * Input.XDot_0[0] * Ttot;
        Q2[n] = U[P+1] / P * Input.XDot_0[1] * Ttot;
        Q3[n] = U[P+1] / P * Input.XDot_0[2] * Ttot;
    }
    // constrain second derivative at starting point
    if (Input.XDDot_0_flag == true) {
        n += 1;
        A(n,0) = U[P+2];
        A(n,1) = -(U[P+1] + U[P+2]);
        A(n,2) = U[P+1];
        Q1[n] = ( pow(U[P+1],2) * U[P+2] / (P*(P-1)) ) * Input.XDDot_0[0] * pow(Ttot,2);
        Q2[n] = ( pow(U[P+1],2) * U[P+2] / (P*(P-1)) ) * Input.XDDot_0[1] * pow(Ttot,2);
        Q3[n] = ( pow(U[P+1],2) * U[P+2] / (P*(P-1)) ) * Input.XDDot_0[2] * pow(Ttot,2);
    }
    n += 1;
    int m = -1;
    int n0 = n;
    Eigen::VectorXd NN(N+K+1), NN1(N+K+1), NN2(N+K+1);
    // constrain waypoints
    for (n = n0; n < N+n0+1; n++) {
        m += 1;
        basisFunction(uk[m], U, N+K+1, P, &NN[0], &NN1[0], &NN2[0]);
        for (int b = 0; b < N+K+1; b++) {
            A(n,b) = NN[b];
        }
        Q1[n] = Input.X1[m];
        Q2[n] = Input.X2[m];
        Q3[n] = Input.X3[m];
    }
    n = N+n0;
    // constrain second derivative at final point
    if (Input.XDDot_N_flag == true) {
        n += 1;
        A(n,N+K-2) = 1 - U[M-P-1];
        A(n,N+K-1) = -(2 - U[M-P-1] - U[M-P-2]);
        A(n,N+K) = 1 - U[M-P-2];
        Q1[n] = ( pow((1-U[M-P-1]),2) * (1-U[M-P-2]) / (P*(P-1)) ) * Input.XDDot_N[0] * pow(Ttot,2);
        Q2[n] = ( pow((1-U[M-P-1]),2) * (1-U[M-P-2]) / (P*(P-1)) ) * Input.XDDot_N[1] * pow(Ttot,2);
        Q3[n] = ( pow((1-U[M-P-1]),2) * (1-U[M-P-2]) / (P*(P-1)) ) * Input.XDDot_N[2] * pow(Ttot,2);
    }
    // constrain first derivative at final point
    if (Input.XDot_N_flag == true) {
        n += 1;
        A(n,N+K-1) = -1;
        A(n,N+K) = 1;
        Q1[n] = (1-U[M-P-1]) / P * Input.XDot_N[0] * Ttot;
        Q2[n] = (1-U[M-P-1]) / P * Input.XDot_N[1] * Ttot;
        Q3[n] = (1-U[M-P-1]) / P * Input.XDot_N[2] * Ttot;
    }

    // solve linear systems
    Eigen::MatrixXd B = A.inverse();
    Eigen::VectorXd C1 = B * Q1;
    Eigen::VectorXd C2 = B * Q2;
    Eigen::VectorXd C3 = B * Q3;

    Output->U = U;
    Output->C1 = C1;
    Output->C2 = C2;
    Output->C3 = C3;

    double dt = 1.0 / (Num - 1);
    double t = 0;
    // store the interpolated trajectory information into Output structure
    Output->T.resize(Num);
    Output->X1.resize(Num);
    Output->X2.resize(Num);
    Output->X3.resize(Num);
    Output->XD1.resize(Num);
    Output->XD2.resize(Num);
    Output->XD3.resize(Num);
    Output->XDD1.resize(Num);
    Output->XDD2.resize(Num);    
    Output->XDD3.resize(Num);
    for (int i = 0; i < Num; i++) {
        basisFunction(t, U, N+K+1, P, &NN[0], &NN1[0], &NN2[0]);
        Output->T[i] = t * Ttot;
        Output->X1[i] = NN.dot(C1);
        Output->X2[i] = NN.dot(C2);
        Output->X3[i] = NN.dot(C3);
        Output->XD1[i]  = NN1.dot(C1) / Ttot;
        Output->XD2[i]  = NN1.dot(C2) / Ttot;
        Output->XD3[i]  = NN1.dot(C3) / Ttot;
        Output->XDD1[i] = NN2.dot(C1) / pow(Ttot,2);
        Output->XDD2[i] = NN2.dot(C2) / pow(Ttot,2);
        Output->XDD3[i] = NN2.dot(C3) / pow(Ttot,2);
        t += dt;
    }

    return;
}

/*! This function takes the Input structure, performs the BSpline LS approximation and outputs the result into Output structure */
void approximate(InputDataSet Input, int Num, int Q, int P, OutputDataSet *Output)
{   
    Output->P = P;

    // N = number of waypoints - 1 
    int N = (int) Input.X1.size() - 1;
    
    // T = time tags; if not specified, it is computed from a cartesian distance assuming a constant velocity norm on average
    Eigen::VectorXd T(N+1);
    double S = 0;
    if (Input.T_flag == true) {
        T = Input.T;
    }
    else {
        T[0] = 0;
        for (int n = 1; n < N+1; n++) {
            T[n] = T[n-1] + pow( (pow(Input.X1[n]-Input.X1[n-1], 2) + pow(Input.X2[n]-Input.X2[n-1], 2) + pow(Input.X3[n]-Input.X3[n-1], 2)), 0.5 );
            S += T[n] - T[n-1];
        }
    }
    if (Input.AvgXDot_flag == true) {
        for (int n = 0; n < N+1; n++) {
            T[n] = T[n] / T[N] * S / Input.AvgXDot;
        }
    }

    double Ttot = T[N];

    // build uk vector: normalized waypoint time tags
    Eigen::VectorXd uk(N+1);
    for (int n = 0; n < N+1; n++) {
        uk[n] = T[n] / Ttot;
    }
    
    // The maximum polynomial order is N + K. If a higher order is requested, print a BSK_ERROR
    if (P > Q) {
        std::cout << "Error in BSpline.approximate: \n the desired polynomial order P can't be higher than the number of control points Q. \n" ;
    }

    // build knot vector U of size Q + P + 2
    Eigen::VectorXd U(Q+P+2);
    double d, alpha;
    int i;
    d = ((double)(N+1)) / ((double)abs(Q-P+1));
    for (int p = 0; p < P+1; p++) {
        U[p] = 0;
    }
    for (int j = 1; j < Q-P+1; j++) {
        i = int(j*d);
        alpha = j*d - i;
        U[P+j] = (1-alpha)*uk[i-1] + alpha*uk[i];
    }
    for (int p = 0; p < P+1; p++) {
        U[Q+p+1] = 1;
    }

    // K = number of endpoint derivatives
    int K = 0;
    if (Input.XDot_0_flag == true) {K += 1;}
    if (Input.XDot_N_flag == true) {K += 1;}
    if (Input.XDDot_0_flag == true) {K += 1;}
    if (Input.XDDot_N_flag == true) {K += 1;}

    // build stiffness matrix MD of size (K+2)x(K+2)
    Eigen::MatrixXd MD(K+2,K+2);
    // build vectors T1, T2, T3 (left hand side of linear system)
    Eigen::VectorXd T1(K+2), T2(K+2), T3(K+2);
    // populate MD with zeros
    for (int n = 0; n < K+2; n++) {
        for (int m = 0; m < K+2; m++) {
            MD(n,m) = 0;
        }
    }
    Eigen::VectorXd NN(Q+1), NN1(Q+1), NN2(Q+1);
    basisFunction(uk[0], U, Q+1, P, &NN[0], &NN1[0], &NN2[0]);
    int n = 0;
    MD(0,0) = NN[0];
    T1[0] = Input.X1[0];
    T2[0] = Input.X2[0];
    T3[0] = Input.X3[0];
    // constrain first derivative at starting point
    if (Input.XDot_0_flag == true) {
        n += 1;
        MD(n,0) = NN1[0];
        MD(n,1) = NN1[1];
        T1[n] = Input.XDot_0[0] * Ttot;
        T2[n] = Input.XDot_0[1] * Ttot;
        T3[n] = Input.XDot_0[2] * Ttot;
    }
    // constrain second derivative at starting point
    if (Input.XDDot_0_flag == true) {
        n += 1;
        MD(n,0) = NN2[0];
        MD(n,1) = NN2[1];
        MD(n,2) = NN2[2];        
        T1[n] = Input.XDDot_0[0] * pow(Ttot,2);
        T2[n] = Input.XDDot_0[1] * pow(Ttot,2);
        T3[n] = Input.XDDot_0[2] * pow(Ttot,2);
    }
    basisFunction(uk[N], U, Q+1, P, &NN[0], &NN1[0], &NN2[0]);
    // constrain second derivative at ending point
    if (Input.XDDot_N_flag == true) {
        n += 1;
        MD(n,K-1) = NN2[Q-2];
        MD(n,K)   = NN2[Q-1];
        MD(n,K+1) = NN2[Q];        
        T1[K-1] = Input.XDDot_N[0] * pow(Ttot,2);
        T2[K]   = Input.XDDot_N[1] * pow(Ttot,2);
        T3[K+1] = Input.XDDot_N[2] * pow(Ttot,2);
    }
    // constrain first derivative at ending point
    if (Input.XDot_N_flag == true) {
        n += 1;
        MD(n,K)   = NN1[Q-1];
        MD(n,K+1) = NN1[Q];
        T1[n] = Input.XDot_N[0] * Ttot;
        T2[n] = Input.XDot_N[1] * Ttot;
        T3[n] = Input.XDot_N[2] * Ttot;
    }
    n += 1;
    MD(n,K+1) = NN[Q];
    T1[n] = Input.X1[N];
    T2[n] = Input.X2[N];
    T3[n] = Input.X3[N];

    // solve linear systems
    Eigen::MatrixXd B = MD.inverse();
    Eigen::VectorXd C1_1 = B * T1;
    Eigen::VectorXd C2_1 = B * T2;
    Eigen::VectorXd C3_1 = B * T3;

    // populate Rk vectors with the base points for LS minimization
    Eigen::VectorXd Rk1(N-1), Rk2(N-1), Rk3(N-1);
    for (int n = 1; n < N; n++) {
        basisFunction(uk[n], U, Q+1, P, &NN[0], &NN1[0], &NN2[0]);
        Rk1[n-1] = Input.X1[n] - NN[0]*C1_1[0] - NN[Q]*C1_1[K+1];
        Rk2[n-1] = Input.X2[n] - NN[0]*C2_1[0] - NN[Q]*C2_1[K+1];
        Rk3[n-1] = Input.X3[n] - NN[0]*C3_1[0] - NN[Q]*C3_1[K+1];
        if (Input.XDot_0_flag == true) {
            Rk1[n-1] -= NN[1]*C1_1[1];
            Rk2[n-1] -= NN[1]*C2_1[1];
            Rk3[n-1] -= NN[1]*C3_1[1];
        }
        if (Input.XDDot_0_flag == true) {
            Rk1[n-1] -= NN[2]*C1_1[2];
            Rk2[n-1] -= NN[2]*C2_1[2];
            Rk3[n-1] -= NN[2]*C3_1[2];
        }
        if (Input.XDDot_N_flag == true) {
            Rk1[n-1] -= NN[Q-2]*C1_1[K-1];
            Rk2[n-1] -= NN[Q-2]*C2_1[K-1];
            Rk3[n-1] -= NN[Q-2]*C3_1[K-1];
        }
        if (Input.XDot_N_flag == true) {
            Rk1[n-1] -= NN[Q-1]*C1_1[K];
            Rk2[n-1] -= NN[Q-1]*C2_1[K];
            Rk3[n-1] -= NN[Q-1]*C3_1[K];
        }
    }
    
    // populate LS matrix ND
    Eigen::MatrixXd ND(N-1,Q-K-1);
    for (int n = 0; n < N-1; n++) {
        basisFunction(uk[1+n], U, Q+1, P, &NN[0], &NN1[0], &NN2[0]);
        int k = 1;
        if (Input.XDot_0_flag == true) {k += 1;}
        if (Input.XDDot_0_flag == true) {k += 1;}
        for (int b = 0; b < Q-K-1; b++) {
            ND(n,b) = NN[k+b];
        }
    }

    // populate weight matrix W
    Eigen::MatrixXd W(N-1,N-1);
    for (int n = 0; n < N-1; n++) {
        for (int m = 0; m < N-1; m++) {
            if (n == m) {
                if (Input.W_flag) {
                    W(n,m) = Input.W[n+1];
                }
                else {
                    W(n,m) = 1;
                }
            }
            else {
                W(n,m) = 0;
            }
        }
    }

    Eigen::VectorXd R1(Q-K-1), R2(Q-K-1), R3(Q-K-1);
    B = ND.transpose() * W;
    R1 = B * Rk1;
    R2 = B * Rk2;
    R3 = B * Rk3;

    // compute LS values R for the control points
    Eigen::MatrixXd NWN(Q-K-1,Q-K-1), NWN_inv(Q-K-1,Q-K-1);
    NWN = B * ND;
    NWN_inv = NWN.inverse();
    Eigen::VectorXd C1_2 = NWN_inv * R1;
    Eigen::VectorXd C2_2 = NWN_inv * R2;
    Eigen::VectorXd C3_2 = NWN_inv * R3;
    
    // build control point vectors C
    Eigen::VectorXd C1(Q+1), C2(Q+1), C3(Q+1);
    n = 0;
    C1[n] = C1_1[n];  C2[n] = C2_1[n];  C3[n] = C3_1[n];
    if (Input.XDot_0_flag == true) {
        n += 1;
        C1[n] = C1_1[n];  C2[n] = C2_1[n];  C3[n] = C3_1[n];
    }
    if (Input.XDDot_0_flag == true) {
        n += 1;
        C1[n] = C1_1[n];  C2[n] = C2_1[n];  C3[n] = C3_1[n];
    }
    for (int q = 0; q < Q-K-1; q++) {
        C1[n+q+1] = C1_2[q];  C2[n+q+1] = C2_2[q];  C3[n+q+1] = C3_2[q];
    }
    if (Input.XDDot_N_flag == true) {
        n += 1;
        C1[Q-K-1+n] = C1_1[n];  C2[Q-K-1+n] = C2_1[n];  C3[Q-K-1+n] = C3_1[n];
    }
    if (Input.XDot_N_flag == true) {
        n += 1;
        C1[Q-K-1+n] = C1_1[n];  C2[Q-K-1+n] = C2_1[n];  C3[Q-K-1+n] = C3_1[n];
    }
    n += 1;
    C1[Q-K-1+n] = C1_1[n];  C2[Q-K-1+n] = C2_1[n];  C3[Q-K-1+n] = C3_1[n];

    Output->U = U;
    Output->C1 = C1;
    Output->C2 = C2;
    Output->C3 = C3;

    double dt = 1.0 / (Num - 1);
    double t = 0;
    // store the interpolated trajectory information into Output structure
    Output->T.resize(Num);
    Output->X1.resize(Num);
    Output->X2.resize(Num);
    Output->X3.resize(Num);
    Output->XD1.resize(Num);
    Output->XD2.resize(Num);
    Output->XD3.resize(Num);
    Output->XDD1.resize(Num);
    Output->XDD2.resize(Num);    
    Output->XDD3.resize(Num);
    for (int i = 0; i < Num; i++) {
        basisFunction(t, U, Q+1, P, &NN[0], &NN1[0], &NN2[0]);
        Output->T[i] = t * Ttot;
        Output->X1[i] = NN.dot(C1);
        Output->X2[i] = NN.dot(C2);
        Output->X3[i] = NN.dot(C3);
        Output->XD1[i]  = NN1.dot(C1) / Ttot;
        Output->XD2[i]  = NN1.dot(C2) / Ttot;
        Output->XD3[i]  = NN1.dot(C3) / Ttot;
        Output->XDD1[i] = NN2.dot(C1) / pow(Ttot,2);
        Output->XDD2[i] = NN2.dot(C2) / pow(Ttot,2);
        Output->XDD3[i] = NN2.dot(C3) / pow(Ttot,2);
        t += dt;
    }

    return;
}

/*! This function calculates the basis functions NN of order P, and derivatives NN1, NN2, for a given time t and knot vector U */
void basisFunction(double t, Eigen::VectorXd U, int I, int P, double *NN, double *NN1, double *NN2)
{   
    Eigen::MatrixXd N(I, P+1);
    Eigen::MatrixXd N1(I, P+1);
    Eigen::MatrixXd N2(I, P+1);
    /* populate matrices with zeros */
    for (int i = 0; i < I; i++) {
        for (int p = 0; p < P+1; p++) {
            N(i,p)  = 0;
            N1(i,p) = 0;
            N2(i,p) = 0;
        }
    }
    /* zero order */
    for (int i = 0; i < I; i++) {
        if ( (t >= U(i)) && (t < U(i+1)) ) {
            N(i,0) = 1;
        }
    }
    if (abs(t-1.0) < 1e-5) {
        N(I-1,0) = 1;
    }
    /* higher order - De Boor formula */
    for (int p = 1; p < P+1; p++) {
        for (int i = 0; i < I; i++) {
            if (U[i+p]-U[i] != 0) {
                N(i,p)  += (t-U[i]) / (U[i+p]-U[i]) * N(i,p-1);
                N1(i,p) += p / (U[i+p]-U[i]) * N(i,p-1);
                N2(i,p) += p / (U[i+p]-U[i]) * N1(i,p-1);
            }
            if (U[i+p+1]-U[i+1] != 0) {
                N(i,p)  += (U[i+p+1]-t) / (U[i+p+1]-U[i+1]) * N(i+1,p-1);
                N1(i,p) -= p / (U[i+p+1]-U[i+1]) * N(i+1,p-1);
                N2(i,p) -= p / (U[i+p+1]-U[i+1]) * N1(i+1,p-1);
            }
        }
    }
    // output result
    for (int i = 0; i < I; i++) {
        *(NN+i)  = N(i,P);
        *(NN1+i) = N1(i,P);
        *(NN2+i) = N2(i,P);
    }

    return;
}
