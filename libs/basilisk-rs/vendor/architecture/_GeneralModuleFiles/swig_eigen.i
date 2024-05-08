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
%module swig_eigen
%{
    #include <Eigen/Dense>
    #include "architecture/utilities/avsEigenMRP.h"
    #include <type_traits>
    #include <limits>
    #include <optional>
    #include <variant>
    #include <string>
    #include <cassert>
    #include <utility>
%}

// Sometimes, the %typemap(out) cannot be used in an 'optimal' way. We are okay
// with this as it is only a speed penalty in some special cases.
#pragma SWIG nowarn=474

%fragment("static_fail", "header") {
template<class>
inline constexpr bool always_false_v = false;
}

%fragment("castPyToC", "header", fragment="static_fail") %{
/*
The following method takes a Python object and tries to convert it to the type T
If this is succesful, the value is returned. If it is unsucessful, an appropriate 
error message is returned.
*/
template<class T>
std::variant<T, std::string> castPyToC(PyObject *input)
{        
    if (PyErr_Occurred()) return "castPyToC was pre-errored!";
    
    if constexpr (std::is_same_v<T, bool>)
    {
        if (!PyBool_Check(input))
        {
            return "Expected value to be a boolean.";
        }

        return PyLong_AsLong(input) == 1;
    }
    else if constexpr (std::is_integral_v<T>)
    {
        // Before 3.10, PyLong_AsLong could call __int__, which silently casts float 
        // objects to int (with loss of data). We want to prevent that, so instead we
        // call PyNumber_Index beforehand, which will call __index__, which does not
        // allow implicit casts to int (and appropriately raises a warning)
        #if PY_MAJOR_VERSION <= 3 && PY_MINOR_VERSION < 10
            input = PyNumber_Index(input);
        #endif

        int overflow{0};
        long long cLongLong = PyLong_AsLongLongAndOverflow(input, &overflow);

        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return "Cannot parse value as an integer.";
        }

        if (overflow != 0)
        {
            return "Integer value is too large or too small (overflows a long long).";
        }

        if (cLongLong > std::numeric_limits<T>::max())
        {
            return "Value is larger than maximum integer limit ("+ std::to_string(std::numeric_limits<T>::max()) +")";
        }
        if (cLongLong < std::numeric_limits<T>::min())
        {
            return "Value is smaller than minimum integer limit ("+ std::to_string(std::numeric_limits<T>::min()) +")";
        }

        return (T) cLongLong;
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        double val = PyFloat_AsDouble(input);
        
        if (PyErr_Occurred())
        {
            PyErr_Clear();
            return "Cannot parse value as a floating point.";
        }

        return (T) val;    
    }
    else
    {
        static_assert(always_false_v<T>, "The stored values are not booleans, integers, or floating points. I don't know how to translate them from Python!");
    }
}
%}

%fragment("castCToPy", "header", fragment="static_fail")
{
template<class T>
PyObject * castCToPy(T input)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        // Using explicit cast to long to prevent compiler warnings
        // is_same_v and is_integral_v should only let long pass anyway
        return PyBool_FromLong((long) input);
    }
    else if constexpr (std::is_integral_v<T>)
    {
        return PyLong_FromLong((long) input);
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        return PyFloat_FromDouble(input);
    }
    else
    {
        static_assert(always_false_v<T>, "The stored values are not booleans, integers, or floating points. I don't know how to translate them to Python!");
    }
}
}

%fragment("rotationConversion", "header")
{
/*
Converts from one rotation type to other (for example, MRPd to Quaterniond).
*/
template<class From, class To>
struct RotationCoversion
{
    inline static To convert(const From& input) {return To{ input };};
};

template<class To>
struct RotationCoversion<Eigen::Vector3d, To>
{
    inline static To convert(const Eigen::Vector3d& input) {return To{(Eigen::MRPd{input}).toRotationMatrix() };};
};

template<class To>
struct RotationCoversion<Eigen::Vector4d, To>
{ 
    inline static To convert(const Eigen::Vector4d& input) {return To{ (Eigen::Quaterniond{ input[0], input[1], input[2], input[3] }).toRotationMatrix() };};
};

template<>
struct RotationCoversion<Eigen::Vector3d, Eigen::MRPd>
{
    inline static Eigen::MRPd convert(const Eigen::Vector3d& input) {return Eigen::MRPd{ input };};
};

template<>
struct RotationCoversion<Eigen::Vector4d, Eigen::Quaterniond>
{
    inline static Eigen::Quaterniond convert(const Eigen::Vector4d& input) {return Eigen::Quaterniond{ input[0], input[1], input[2], input[3] };};
};

}

%fragment("getInputSize", "header")
{
/*
This function returns the size of the input.
If the input is not a sequence, then an empty optional is returned.
Otherwise, the number of rows and columns are returned in the pair.
The number of columns are obtained from the first element of the input sequence.
(i.e. ragged nested sequences are not checked)
*/
std::optional<std::pair<Py_ssize_t, Py_ssize_t>> getInputSize(PyObject *input)
{
    // Vectors and matrices must come from python sequences
    if(!PySequence_Check(input)) {
        return {};
    }

    Py_ssize_t numberRows = PySequence_Length(input);

    PyObject *firstItem = PySequence_GetItem(input, 0);
    Py_ssize_t numberColumns = PySequence_Check(firstItem) ? PySequence_Length(firstItem) : 1;
    Py_DECREF(firstItem);
    return {{numberRows, numberColumns}};
}
}

%fragment("checkPyObjectIsMatrixLike", "header", fragment="getInputSize") {

/*
This method returns an empty optional only if the given input is like the template type T.
Otherwise, the returned optional contains a relevant error message.
The size of the input is compared to the expected size of T (where dynamically
sized matrix T allow any number of rows/columns). Moreover, an error is raised for
ragged nested sequences (rows with different numbers of elements).
*/
template<class T>
std::optional<std::string> checkPyObjectIsMatrixLike(PyObject *input)
{
    auto maybeSize = getInputSize(input);

    if (!maybeSize)
    {
        return "Input is not a sequence";
    }

    auto [numberRows, numberColumns] = maybeSize.value(); 

    if (T::RowsAtCompileTime != -1 && T::RowsAtCompileTime != numberRows)
    {
        std::string errorMsg = "Input does not have the correct number of rows. Expected " 
            + std::to_string(T::RowsAtCompileTime) + " but found " + std::to_string(numberRows);
        return errorMsg;
    }

    if (T::ColsAtCompileTime != -1 && T::ColsAtCompileTime != numberColumns)
    {
        std::string errorMsg = "Input does not have the correct number of columns. Expected " 
            + std::to_string(T::ColsAtCompileTime) + " but found " + std::to_string(numberColumns);
        return errorMsg;
    }

    for(Py_ssize_t row=0; row<numberRows; row++)
    {
        PyObject *rowPyObj = PySequence_GetItem(input, row);
        Py_ssize_t localNumberColumns = PySequence_Check(rowPyObj) ? PySequence_Length(rowPyObj) : 1;
        if (localNumberColumns != numberColumns)
        {
            return "All rows must be the same length! Row " + std::to_string(row) + " is not.";
        }
    }
    
    return {};
}
}

%fragment("pyObjToEigenMatrix", "header", fragment="castPyToC", fragment="getInputSize", fragment="checkPyObjectIsMatrixLike") {

/*
Creates a new Eigen::Matrix of the templated type T with the given 'input'.
Might return a default-initialized T if an error occurred during translation.
In that case, disambiguate using PyErr_Occurred().
*/
template<class T>
T pyObjToEigenMatrix(PyObject *input)
{
    using ScalarType = typename T::Scalar;

    auto errorMsg = checkPyObjectIsMatrixLike<T>(input);
    if (errorMsg)
    {
        PyErr_SetString(PyExc_ValueError,errorMsg.value().c_str());
        return {};
    }
    // After the previous check, we can assume the input is a sequence
    // that has correct size.

    T result;

    auto [numberRows, numberColumns] = getInputSize(input).value();
    
    // Resize can be called even for non-dynamic matrices as long as the
    // fixed-sizes do not change. 
    result.resize(numberRows, numberColumns);

    for(Py_ssize_t row=0; row<numberRows; row++)
    {
        PyObject *rowPyObj = PySequence_GetItem(input, row);
        bool rowPyObjIsSequence = PySequence_Check(rowPyObj);

        for(Py_ssize_t col=0; col<numberColumns; col++)
        {
            // rowPyObj can be either a length 1 sequence or the value directly
            std::variant<ScalarType, std::string> valueOrErrorMsg;
            if (rowPyObjIsSequence)
            {
                PyObject *rowColPyObj = PySequence_GetItem(rowPyObj, col);
                valueOrErrorMsg = castPyToC<ScalarType>(rowColPyObj);
                Py_DECREF(rowColPyObj);
            }
            else
            {
                valueOrErrorMsg = castPyToC<ScalarType>(rowPyObj);
            } 

            if (std::holds_alternative<std::string>(valueOrErrorMsg))
            {
                PyErr_SetString(PyExc_ValueError, (
                    "Row " + std::to_string(row) + ", Column " + std::to_string(col) +": "
                    + std::get<std::string>(valueOrErrorMsg)
                ).c_str());
                
                Py_DECREF(rowPyObj);

                return {};
            }

            result(row, col) = std::get<typename T::Scalar>(valueOrErrorMsg);
        }

        Py_DECREF(rowPyObj);
    }

    return result;
}
}

%fragment("pyObjToRotation", "header", fragment="pyObjToEigenMatrix", fragment="getInputSize", fragment="rotationConversion")
{
/*
Creates a new rotation of the templated type T with the given 'input'.
Translation might fail, in which case the error flag will be set (disambiguate 
using PyErr_Occurred()). If this happens, the returned value is undefined.
*/
template<class T>
T pyObjToRotation(PyObject *input)
{
    auto maybeSize = getInputSize(input);

    if (!maybeSize) // Not even a sequence
    {
        std::string errorMsg = "Input is not a sequence";
        PyErr_SetString(PyExc_ValueError,errorMsg.c_str());
        return {};
    }

    auto [numberRows, numberColumns] = maybeSize.value(); 

    if (numberRows == 3 && numberColumns == 1)
    {
        return RotationCoversion<Eigen::Vector3d, T>::convert(pyObjToEigenMatrix<Eigen::Vector3d>(input));
    }
    else if (numberRows == 4 && numberColumns == 1)
    {
        return RotationCoversion<Eigen::Vector4d, T>::convert(pyObjToEigenMatrix<Eigen::Vector4d>(input));
    }
    else if (numberRows == 3 && numberColumns == 3)
    {
        return RotationCoversion<Eigen::Matrix3d, T>::convert(pyObjToEigenMatrix<Eigen::Matrix3d>(input));
    }
    else
    {
        std::string errorMsg = "Unknown rotation dimensions: " 
            + std::to_string(numberRows) + "x" + std::to_string(numberColumns)
            + ". Expected dimensions 3x1, 4x1, or 3x3.";
        PyErr_SetString(PyExc_ValueError, errorMsg.c_str());
        return {};
    }
}
}

%fragment("fillPyObjList", "header", fragment="castCToPy") {

/*
Copies the values in 'value' to the list PyObject in 'input'
*/
template<class T>
void fillPyObjList(PyObject *input, const T& value)
{
    for(auto i=0; i<value.innerSize(); i++)
    {
        PyObject *locRow = PyList_New(0);
        for(auto j=0; j<value.outerSize(); j++)
        {            
            auto toAppend = castCToPy<typename T::Scalar>(value(i,j));
            PyList_Append(locRow, toAppend);
            Py_DECREF(toAppend);
        }
        PyList_Append(input, locRow);
        Py_DECREF(locRow);
    }
}

// Eigen::MRPd and Eigen::Quaterniond need to be converted first to Eigen::Matrix
template<>
void fillPyObjList<Eigen::MRPd>(PyObject *input, const Eigen::MRPd& value)
{
    return fillPyObjList(input, value.vec());
}

template<>
void fillPyObjList<Eigen::Quaterniond>(PyObject *input, const Eigen::Quaterniond& value)
{
    return fillPyObjList<Eigen::Vector4d>(input, {value.w(), value.x(), value.y(), value.z()});
}

}

%define EIGEN_MAT_WRAP(type, typeCheckPrecedence)

%typemap(in, fragment="pyObjToEigenMatrix") type {
    $1 = pyObjToEigenMatrix<type>($input);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(memberin) type {
    $1 = std::move($input);
}

%typemap(in, fragment="pyObjToEigenMatrix") type & {
    $1 = new type;
    *$1 = pyObjToEigenMatrix<type>($input);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(freearg) type & {
    delete $1;
}

%typemap(typecheck, fragment="checkPyObjectIsMatrixLike", precedence= ## typeCheckPrecedence) type {
    // PyErr_Fetch and PyErr_Restore preserve/restore the error status before this function
    // We use the error flag to check whether conversion is valid, but we do not want to
    // alter the previous error status of the program
    PyObject *ty, *value, *traceback;
    PyErr_Fetch(&ty, &value, &traceback);

    pyObjToEigenMatrix<type>($input);
    $1 = ! PyErr_Occurred();

    PyErr_Restore(ty, value, traceback);
}

%typemap(out, optimal="1", fragment="fillPyObjList") type {
    $result = PyList_New(0);
    fillPyObjList<type>($result, $1);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(out, fragment="fillPyObjList") type * {
    if(!($1))
    {
        $result = SWIG_Py_Void();
    }
    else
    {
        $result = PyList_New(0);
        fillPyObjList<type>($result, *$1);
    }
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(typecheck) type & = type;
%typemap(typecheck) type && = type;

%typemap(in) type && = type &;
%typemap(freearg) type && = type &;

%enddef

%define EIGEN_ROT_WRAP(type, typeCheckPrecedence)

%typemap(in, fragment="pyObjToRotation") type {
    $1 = pyObjToRotation<type>($input);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(in, fragment="pyObjToRotation") type & {
    $1 = new type;
    *$1 = pyObjToRotation<type>($input);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(freearg) type & {
    delete $1;
}

%typemap(typecheck, fragment="getInputSize", precedence= ## typeCheckPrecedence) type {
    auto maybeSize = getInputSize($input);

    if (!maybeSize)
    {
        $1 = 0;
    }
    else
    {
        auto [numberRows, numberColumns] = maybeSize.value();
        $1 = (
            (numberRows == 3 && numberColumns == 1) ||
            (numberRows == 4 && numberColumns == 1) ||
            (numberRows == 3 && numberColumns == 3)
        ) ? 1 : 0;
    }
}

%typemap(out, optimal="1", fragment="fillPyObjList") type {
    $result = PyList_New(0);
    fillPyObjList($result, $1);
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(out, fragment="fillPyObjList") type * {
    if(!($1))
    {
        $result = SWIG_Py_Void();
    }
    else
    {
        $result = PyList_New(0);
        fillPyObjList($result, *$1);
    }
    if (PyErr_Occurred()) SWIG_fail;
}

%typemap(typecheck) type & = type;
%typemap(typecheck) type && = type;

%typemap(in) type && = type &;
%typemap(freearg) type && = type &;

%enddef

// Second value is the precedence for overloads. If two methods are overloaded
// with different Eigen inputs, then the precedence determines which ones are
// attempted first.
//
// Fully fixed-size types should have precendence 155 < precedence < 160
// Types partially fixed-size should have precedence 160 < precedence < 165
// Fully dynamically sized should have precedence 165 < precendence < 170
//
// Boolean matrices should have lower precendence than integer matrices, 
// which should have lower precendence than double matrices.
//
// For reference, std::array has precendence 155, and std::vector has precendence 160
//
// If we were to pass a list of three integers from python to an overloaded method 'foo',
// this would be the order in which they would be chosen (from first to last):
//      foo(std::array<int>)
//      foo(Vector3i)
//      foo(Vector3d)
//      foo(std::vector<int>)
//      foo(VectorXi)
//      foo(VectorXd)
//      foo(MatrixXi)
//      foo(MatrixXd)

EIGEN_MAT_WRAP(Eigen::Vector3i, 158)
EIGEN_MAT_WRAP(Eigen::Vector3d, 159)
EIGEN_MAT_WRAP(Eigen::Matrix3d, 159)
EIGEN_MAT_WRAP(Eigen::MatrixX3i,163)
EIGEN_MAT_WRAP(Eigen::VectorXi, 163)
EIGEN_MAT_WRAP(Eigen::VectorXd, 164)
EIGEN_MAT_WRAP(Eigen::MatrixX3d,164)
EIGEN_MAT_WRAP(Eigen::MatrixXi, 169)
EIGEN_MAT_WRAP(Eigen::MatrixXd, 169)

// Rotation wrappers work so that they can be set from Python on any
// of the three ways we have of representing rotations: MRPs (3x1 vector),
// quaternions (4x1 vector), or rotation matrix (3x3 vector).
// 
// Their return type, however, is always an MRPd.
//
// The precedence should be that of fixed-size double eigen matrices (159). 
EIGEN_ROT_WRAP(Eigen::MRPd       , 159)
EIGEN_ROT_WRAP(Eigen::Quaterniond, 159)
