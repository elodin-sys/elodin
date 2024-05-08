# ISC License
#
# Copyright (c) 2023, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
import pytest
import numpy as np

from Basilisk.architecture.swigEigenCheck import SwigEigenTestClass
from Basilisk.utilities import RigidBodyKinematics as rbk


def getData(type: str):
    if type.lower() == "mrpd":
        return [0, 0.5, 0], [[0], [0.5], [0]]
    if type.lower() == "quaterniond":
        return [0, 0, 1, 0], [[0], [0], [1], [0]]

    scalar = type[-1]
    isMatrix = "matrix" in type.lower()
    dimensions = type[:-1].lower().replace("matrix", "").replace("vector", "")

    if len(dimensions) == 1:
        if isMatrix:
            dimensions += dimensions
        else:
            dimensions += "1"

    size = [7 if dim == "x" else int(dim) for dim in dimensions]

    i = 0.5
    inputData = []
    outputData = []

    def lambda_gen(i):
        return int(i) if scalar == "i" else i

    for _ in range(size[0]):
        if size[1] == 1:
            val = lambda_gen(i)
            inputData.append(val)
            outputData.append([val])
            i += 1
        else:
            val = lambda_gen(i)
            inputData.append([])
            outputData.append([])
            for _ in range(size[1]):
                inputData[-1].append(val)
                outputData[-1].append(val)
                i += 1

    return inputData, outputData


def getRotationData(input: str, output: str):
    if input == "mrpd":
        input_data = [0, 0.5, 0]
        if output == "mrpd":
            output_data = np.array([[0], [0.5], [0]])
        elif output == "quaterniond":
            output_data = rbk.MRP2EP([0, 0.5, 0]).reshape(4, 1)
        else:
            raise ValueError(f"Unknown output type {output}")

    elif input == "quaterniond":
        input_data = np.array([[0], [0.5], [0], [0.2]])
        input_data /= np.linalg.norm(input_data)
        if output == "mrpd":
            output_data = rbk.EP2MRP(input_data).reshape(3, 1)
        elif output == "quaterniond":
            output_data = input_data
        else:
            raise ValueError(f"Unknown output type {output}")

    elif input == "rotationMatrix":
        input_data = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        if output == "mrpd":
            output_data = rbk.C2MRP(input_data).reshape(3, 1)
        elif output == "quaterniond":
            output_data = rbk.C2EP(input_data).reshape(4, 1)
        else:
            raise ValueError(f"Unknown output type {output}")

    else:
        raise ValueError(f"Unknown input type {input}")

    return input_data, output_data


@pytest.mark.parametrize(
    "attr",
    [
        "vector3i",
        "vector3d",
        "matrix3d",
        "matrixX3i",
        "vectorXi",
        "vectorXd",
        "matrixX3d",
        "matrixXi",
        "matrixXd",
    ],
)
@pytest.mark.parametrize("useNumpy", [False, True])
def test_simpleInputOutput(attr: str, useNumpy: bool):
    testClass = SwigEigenTestClass()

    inputData, outputData = getData(attr)

    setattr(testClass, attr, np.array(inputData) if useNumpy else inputData)
    retrieved = getattr(testClass, attr)

    assert outputData == retrieved, f"{attr} {retrieved}"


@pytest.mark.parametrize(
    "keyword,attr,data",
    [
        ("not a sequence", "vector3d", 1),
        ("number of rows", "vector3d", [1, 2]),
        ("number of rows", "vector3d", [1, 2, 3, 4]),
        ("number of columns", "vector3d", [[1, 2], [1, 2], [1, 2]]),
        ("number of columns", "matrixX3d", [[1, 2, 3, 4]]),
        ("rows must be the same length", "matrixXd", [[1, 2, 3], [1, 2, 3], [1, 2]]),
        ("Unknown rotation dimensions", "mrpd", [1, 2]),
        ("Unknown rotation dimensions", "mrpd", [1, 2, 3, 4, 5]),
        ("Unknown rotation dimensions", "mrpd", [[1, 2], [1, 2], [1, 2]]),
        ("parse value as an integer", "vector3i", [1.0, 0, 0]),
        ("parse value as an integer", "vector3i", [1, 0, 1.0]),
        ("parse value as a floating point", "vector3d", [1.0, 0, "0"]),
        ("overflow", "vector3i", [int(1e99), 0, 0]),
        # The size of int is implementation specific, so the following two tests could
        # fail in some implementations
        # ("larger than maximum", "vector3i", [2147483647+1, 0, 0]),
        # ("smaller than minimum", "vector3i", [-2147483647-2, 0, 0]),
    ],
)
def test_illegal(keyword: str, attr: str, data):
    testClass = SwigEigenTestClass()

    with pytest.raises(ValueError, match=rf".*{keyword}.*"):
        setattr(testClass, attr, data)


@pytest.mark.parametrize("input", ["mrpd", "quaterniond", "rotationMatrix"])
@pytest.mark.parametrize("output", ["mrpd", "quaterniond"])
def test_rotation(input: str, output: str):
    testClass = SwigEigenTestClass()

    inpData, outData = getRotationData(input, output)

    setattr(testClass, output, inpData)
    retrieved = getattr(testClass, output)

    np.testing.assert_almost_equal(outData, np.array(retrieved), 8)


# The following test asserts that overloading precedence is respected
# for all overloaded functions ending in "Precedence" in SwigEigenTestClass
@pytest.mark.parametrize(
    "functionName",
    [
        attr
        for attr in dir(SwigEigenTestClass)
        if attr.startswith("test") and "Precedence" in attr
    ],
)
def test_precedence(functionName: str):
    testClass = SwigEigenTestClass()
    testData = [1, 2, 3]

    assert getattr(testClass, functionName)(testData)


# The following test asserts that the non-precedent overloaded functions
# can be reached if the input is tuned so that the function that normally
# takes precedence can no longer accept the input, and so the second function
# is called. (For example, [1, 2] is not a valid input for Vector3d, but it
# is for std::vector<int>)
@pytest.mark.parametrize(
    "functionName,testData",
    [
        ("testArrayPrecedence", [1, [2], 3]),
        ("testVector3iPrecedence", [1.0, 2.0, 3.0]),
        ("testVector3dPrecedence", [1, 2]),
        ("testVectorXiPrecedence", [1.0, 2.0]),
        ("testVectorXdPrecedence", [[1, 2], [3, 4]]),
        ("testMatrixXiPrecedence", [[1, 2], [3.0, 4]]),
        ("testStdVectorPrecedence", [[1], 2, 3, 4, 5]),
    ],
)
def test_antiPrecedence(functionName: str, testData):
    testClass = SwigEigenTestClass()

    assert not getattr(testClass, functionName)(testData)


def test_nullPointerReturn():
    testClass = SwigEigenTestClass()
    data = [[1.0], [2.0], [3.0]]
    testClass.vector3d = data

    # returnVector3dPointer will return a nullptr if True is passed
    # otherwise, it returns vector3d

    assert testClass.returnVector3dPointer(True) is None
    assert testClass.returnVector3dPointer(False) == data


# Check that we can call functions with any qualifiers in the signature
#   void qualifierTestValue   (      Eigen::Vector3d  )
#   void qualifierTestConstRef(const Eigen::Vector3d& )
#   void qualifierTestConst   (const Eigen::Vector3d  )
#   void qualifierTestRef     (      Eigen::Vector3d& )
#   void qualifierTestRvalue  (      Eigen::Vector3d&&)
@pytest.mark.parametrize("qualifier", ["Value", "ConstRef", "Const", "Ref", "Rvalue"])
@pytest.mark.parametrize(
    "attr",
    [
        "vector3i",
        "vector3d",
        "matrix3d",
        "matrixX3i",
        "vectorXi",
        "vectorXd",
        "matrixX3d",
        "matrixXi",
        "matrixXd",
        "quaterniond",
        "mrpd",
    ],
)
def test_qualifiers(qualifier: str, attr: str):
    testClass = SwigEigenTestClass()

    functionName = f"qualifierTest{attr}{qualifier}"

    inputData, outputData = getData(attr)

    try:
        getattr(testClass, functionName)(inputData)
    except TypeError:
        assert False, "Cannot call method with qualifiers from" + functionName

    retrieved = getattr(testClass, attr)
    assert outputData == retrieved
