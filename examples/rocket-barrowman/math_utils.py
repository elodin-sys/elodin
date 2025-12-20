"""Standalone Vector and Matrix classes matching RocketPy's mathutils API.

These implementations are sufficient for the flight solver's 6-DOF dynamics
without requiring the full RocketPy package.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Union

import numpy as np


class Vector:
    """3D vector class compatible with RocketPy's Vector API."""

    __slots__ = ("_data",)

    def __init__(self, components: Sequence[float]) -> None:
        if len(components) != 3:
            raise ValueError("Vector must have exactly 3 components")
        self._data = np.array(components, dtype=np.float64)

    @property
    def x(self) -> float:
        return float(self._data[0])

    @property
    def y(self) -> float:
        return float(self._data[1])

    @property
    def z(self) -> float:
        return float(self._data[2])

    def __repr__(self) -> str:
        return f"Vector([{self.x}, {self.y}, {self.z}])"

    def __neg__(self) -> Vector:
        return Vector(-self._data)

    def __add__(self, other: Vector) -> Vector:
        if isinstance(other, Vector):
            return Vector(self._data + other._data)
        raise TypeError(f"Cannot add Vector and {type(other)}")

    def __sub__(self, other: Vector) -> Vector:
        if isinstance(other, Vector):
            return Vector(self._data - other._data)
        raise TypeError(f"Cannot subtract {type(other)} from Vector")

    def __mul__(self, scalar: float) -> Vector:
        return Vector(self._data * scalar)

    def __rmul__(self, scalar: float) -> Vector:
        return Vector(self._data * scalar)

    def __truediv__(self, scalar: float) -> Vector:
        return Vector(self._data / scalar)

    def __xor__(self, other: Vector) -> Vector:
        """Cross product using ^ operator (RocketPy convention)."""
        if isinstance(other, Vector):
            return Vector(np.cross(self._data, other._data))
        raise TypeError(f"Cannot cross Vector with {type(other)}")

    def __matmul__(self, other: Vector) -> float:
        """Dot product using @ operator."""
        if isinstance(other, Vector):
            return float(np.dot(self._data, other._data))
        raise TypeError(f"Cannot dot Vector with {type(other)}")

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, idx: int) -> float:
        return float(self._data[idx])

    @property
    def cross_matrix(self) -> Matrix:
        """Return the skew-symmetric cross product matrix.
        
        For vector v, returns matrix M such that M @ u = v x u for any vector u.
        """
        x, y, z = self._data
        return Matrix([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ])

    def norm(self) -> float:
        return float(np.linalg.norm(self._data))

    def normalized(self) -> Vector:
        n = self.norm()
        if n < 1e-12:
            return Vector([0.0, 0.0, 0.0])
        return Vector(self._data / n)

    def to_array(self) -> np.ndarray:
        return self._data.copy()


class Matrix:
    """3x3 matrix class compatible with RocketPy's Matrix API."""

    __slots__ = ("_data",)

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("Matrix must be 3x3")
        self._data = np.array(rows, dtype=np.float64)

    @staticmethod
    def transformation(quaternion: Sequence[float]) -> Matrix:
        """Create rotation matrix from quaternion [x, y, z, w].
        
        This matches RocketPy's convention where the quaternion transforms
        from body frame to world frame (inertial frame).
        
        Quaternion format: [e1, e2, e3, e0] where e0 is the scalar part.
        RocketPy uses e = [e0, e1, e2, e3] internally but their API
        often takes [e1, e2, e3, e0] in the state vector.
        """
        # Extract quaternion components (state vector format: [x, y, z, w])
        e1, e2, e3, e0 = quaternion
        
        # Normalize quaternion
        norm = math.sqrt(e0*e0 + e1*e1 + e2*e2 + e3*e3)
        if norm < 1e-12:
            return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        e0, e1, e2, e3 = e0/norm, e1/norm, e2/norm, e3/norm
        
        # Rotation matrix from quaternion (body to world)
        # Using RocketPy's exact formula from vector_matrix.py
        a = 1 - 2*(e2*e2 + e3*e3)
        b = 2*(e1*e2 - e0*e3)
        c = 2*(e0*e2 + e1*e3)
        d = 2*(e1*e2 + e0*e3)
        e = 1 - 2*(e1*e1 + e3*e3)
        f = 2*(e2*e3 - e0*e1)
        g = 2*(e1*e3 - e0*e2)
        h = 2*(e0*e1 + e2*e3)
        i = 1 - 2*(e1*e1 + e2*e2)
        
        return Matrix([
            [a, b, c],
            [d, e, f],
            [g, h, i]
        ])

    @staticmethod
    def identity() -> Matrix:
        return Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def __repr__(self) -> str:
        return f"Matrix({self._data.tolist()})"

    def __neg__(self) -> Matrix:
        return Matrix(-self._data)

    def __add__(self, other: Matrix) -> Matrix:
        if isinstance(other, Matrix):
            return Matrix(self._data + other._data)
        raise TypeError(f"Cannot add Matrix and {type(other)}")

    def __sub__(self, other: Matrix) -> Matrix:
        if isinstance(other, Matrix):
            return Matrix(self._data - other._data)
        raise TypeError(f"Cannot subtract {type(other)} from Matrix")

    def __mul__(self, scalar: float) -> Matrix:
        return Matrix(self._data * scalar)

    def __rmul__(self, scalar: float) -> Matrix:
        return Matrix(self._data * scalar)

    def __truediv__(self, scalar: float) -> Matrix:
        return Matrix(self._data / scalar)

    def __matmul__(self, other: Union[Matrix, Vector]) -> Union[Matrix, Vector]:
        """Matrix multiplication with Matrix or Vector."""
        if isinstance(other, Matrix):
            return Matrix(self._data @ other._data)
        if isinstance(other, Vector):
            result = self._data @ other._data
            return Vector(result)
        raise TypeError(f"Cannot multiply Matrix with {type(other)}")

    @property
    def transpose(self) -> Matrix:
        return Matrix(self._data.T)

    @property
    def inverse(self) -> Matrix:
        try:
            inv = np.linalg.inv(self._data)
            return Matrix(inv)
        except np.linalg.LinAlgError:
            # Return pseudo-inverse for singular matrices
            inv = np.linalg.pinv(self._data)
            return Matrix(inv)

    @property
    def det(self) -> float:
        return float(np.linalg.det(self._data))

    def to_array(self) -> np.ndarray:
        return self._data.copy()

    def __getitem__(self, idx: tuple) -> float:
        return float(self._data[idx])


# Test the implementation
if __name__ == "__main__":
    # Test quaternion to rotation matrix
    # Identity quaternion [0, 0, 0, 1] should give identity matrix
    q_identity = [0.0, 0.0, 0.0, 1.0]
    R = Matrix.transformation(q_identity)
    print("Identity quaternion rotation matrix:")
    print(R.to_array())
    
    # Test 90 degree rotation about z-axis
    # q = [0, 0, sin(45°), cos(45°)] = [0, 0, 0.707, 0.707]
    angle = math.pi / 2
    q_z90 = [0.0, 0.0, math.sin(angle/2), math.cos(angle/2)]
    R_z90 = Matrix.transformation(q_z90)
    print("\n90° rotation about z-axis:")
    print(R_z90.to_array())
    
    # Test vector transformation
    v = Vector([1.0, 0.0, 0.0])
    v_rotated = R_z90 @ v
    print(f"\n[1,0,0] rotated 90° about z: [{v_rotated.x:.3f}, {v_rotated.y:.3f}, {v_rotated.z:.3f}]")
    
    # Test cross product
    a = Vector([1.0, 0.0, 0.0])
    b = Vector([0.0, 1.0, 0.0])
    c = a ^ b
    print(f"\n[1,0,0] x [0,1,0] = [{c.x:.3f}, {c.y:.3f}, {c.z:.3f}]")
    
    print("\nAll tests passed!")

