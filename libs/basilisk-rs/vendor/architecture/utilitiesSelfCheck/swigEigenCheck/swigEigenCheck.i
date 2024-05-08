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

%module swigEigenCheck
#pragma SWIG nowarn=509
%{
    #include <vector>
    #include <array>
%}

%include "std_vector.i"
%include "std_array.i"

%template(IntVector) std::vector<int>;
%template(IntArray3) std::array<int, 3>;

%include "swig_eigen.i"

%define ADD_QUALIFIER_FUNCTIONS(type, memberName)
    void qualifierTest ## memberName ## Value   (      Eigen:: ## type   input) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## ConstRef(const Eigen:: ## type&  input) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Const   (const Eigen:: ## type  input) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Ref     (      Eigen:: ## type&  input) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Rvalue  (      Eigen:: ## type&& input) {this-> ## memberName = input;};

    // We overload the functions to test that the typecheck maps are working properly
    void qualifierTest ## memberName ## Value   (      Eigen:: ## type   input, bool) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## ConstRef(const Eigen:: ## type&  input, bool) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Const   (const Eigen:: ## type  input, bool) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Ref     (      Eigen:: ## type&  input, bool) {this-> ## memberName = input;};
    void qualifierTest ## memberName ## Rvalue  (      Eigen:: ## type&& input, bool) {this-> ## memberName = input;};
%enddef

%inline {

struct SwigEigenTestClass
{
    Eigen::Vector3i vector3i;
    Eigen::Vector3d vector3d;
    Eigen::Matrix3d matrix3d;
    Eigen::MatrixX3i matrixX3i;
    Eigen::VectorXi vectorXi;
    Eigen::VectorXd vectorXd;
    Eigen::MatrixX3d matrixX3d;
    Eigen::MatrixXi matrixXi;
    Eigen::MatrixXd matrixXd;

    Eigen::MRPd mrpd;
    Eigen::Quaterniond quaterniond;

    bool testArrayPrecedence(std::array<int, 3>) {return true;}
    bool testArrayPrecedence(Eigen::Vector3i)    {return false;}

    bool testVector3iPrecedence(Eigen::Vector3i) {return true;}
    bool testVector3iPrecedence(Eigen::Vector3d) {return false;}

    bool testVector3dPrecedence(Eigen::Vector3d)  {return true;}
    bool testVector3dPrecedence(std::vector<int>) {return false;}

    bool testStdVectorPrecedence(std::vector<int>) {return true;}
    bool testStdVectorPrecedence(Eigen::VectorXi)  {return false;}

    bool testVectorXiPrecedence(Eigen::VectorXi) {return true;}
    bool testVectorXiPrecedence(Eigen::VectorXd)  {return false;}

    bool testVectorXdPrecedence(Eigen::VectorXd) {return true;}
    bool testVectorXdPrecedence(Eigen::MatrixXi)  {return false;}

    bool testMatrixXiPrecedence(Eigen::MatrixXi) {return true;}
    bool testMatrixXiPrecedence(Eigen::MatrixXd) {return false;}

    Eigen::Vector3d* returnVector3dPointer(bool returnNull) {return returnNull ? nullptr : &this->vector3d;}

    ADD_QUALIFIER_FUNCTIONS(Vector3i, vector3i)
    ADD_QUALIFIER_FUNCTIONS(Vector3d, vector3d)
    ADD_QUALIFIER_FUNCTIONS(Matrix3d, matrix3d)
    ADD_QUALIFIER_FUNCTIONS(MatrixX3i, matrixX3i)
    ADD_QUALIFIER_FUNCTIONS(VectorXi, vectorXi)
    ADD_QUALIFIER_FUNCTIONS(VectorXd, vectorXd)
    ADD_QUALIFIER_FUNCTIONS(MatrixX3d, matrixX3d)
    ADD_QUALIFIER_FUNCTIONS(MatrixXi, matrixXi)
    ADD_QUALIFIER_FUNCTIONS(MatrixXd, matrixXd)    
    ADD_QUALIFIER_FUNCTIONS(MRPd, mrpd)  
    ADD_QUALIFIER_FUNCTIONS(Quaterniond, quaterniond)  
};

}
