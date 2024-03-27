use std::ops::{Add, Mul};

use nox_ecs::nox::{self, FromOp, IntoOp, Noxpr, Scalar, Vector};
use pyo3::{prelude::*, types::PyTuple};

use crate::Error;

#[pyclass]
pub struct SpatialTransform {
    inner: nox::SpatialTransform<f64>,
}

impl From<nox::SpatialTransform<f64>> for SpatialTransform {
    fn from(inner: nox::SpatialTransform<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl SpatialTransform {
    #[new]
    fn new(arr: PyObject) -> Self {
        nox::SpatialTransform::from_op(Noxpr::jax(arr)).into()
    }

    #[staticmethod]
    fn from_linear(arr: PyObject) -> Self {
        nox::SpatialTransform::from_linear(Vector::from_op(Noxpr::jax(arr))).into()
    }

    #[staticmethod]
    fn from_angular(quat: Quaternion) -> Self {
        nox::SpatialTransform::from_angular(quat.inner).into()
    }

    #[staticmethod]
    fn from_axis_angle(axis: PyObject, angle: PyObject) -> Self {
        nox::SpatialTransform::from_axis_angle(
            Vector::from_op(Noxpr::jax(axis)),
            Scalar::from_op(Noxpr::jax(angle)),
        )
        .into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }
    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialTransform::from_op(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialTransform::from_op(Noxpr::jax(jax)).into()
    }

    #[staticmethod]
    pub fn zero() -> Self {
        nox::SpatialTransform::zero().into()
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.linear().into_op().to_jax()?)
    }

    fn angular(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.angular(),
        }
    }

    fn asarray(&self) -> Result<PyObject, Error> {
        Ok(self.inner.clone().into_op().to_jax()?)
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [7]).into())
    }
}

#[pyclass]
pub struct SpatialMotion {
    inner: nox::SpatialMotion<f64>,
}

impl From<nox::SpatialMotion<f64>> for SpatialMotion {
    fn from(inner: nox::SpatialMotion<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl SpatialMotion {
    #[new]
    fn new(angular: PyObject, linear: PyObject) -> Self {
        let angular = Vector::from_op(Noxpr::jax(angular));
        let linear = Vector::from_op(Noxpr::jax(linear));
        nox::SpatialMotion::new(angular, linear).into()
    }

    #[staticmethod]
    fn from_linear(arr: PyObject) -> Self {
        nox::SpatialMotion::from_linear(Vector::from_op(Noxpr::jax(arr))).into()
    }

    #[staticmethod]
    fn from_angular(arr: PyObject) -> Self {
        nox::SpatialMotion::from_angular(Vector::from_op(Noxpr::jax(arr))).into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }
    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialMotion::from_op(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialMotion::from_op(Noxpr::jax(jax)).into()
    }

    #[staticmethod]
    pub fn zero() -> Self {
        nox::SpatialMotion::zero().into()
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.linear().into_op().to_jax()?)
    }

    fn angular(&self) -> Result<PyObject, Error> {
        Ok(self.inner.angular().into_op().to_jax()?)
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [6]).into())
    }
}

#[pyclass]
pub struct SpatialForce {
    inner: nox::SpatialForce<f64>,
}

impl From<nox::SpatialForce<f64>> for SpatialForce {
    fn from(inner: nox::SpatialForce<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl SpatialForce {
    #[new]
    fn new(arr: PyObject) -> Self {
        nox::SpatialForce::from_op(Noxpr::jax(arr)).into()
    }

    #[staticmethod]
    fn zero() -> Self {
        nox::SpatialForce::zero().into()
    }

    #[staticmethod]
    fn from_linear(arr: PyObject) -> Self {
        nox::SpatialForce::from_linear(Vector::from_op(Noxpr::jax(arr))).into()
    }

    #[staticmethod]
    fn from_torque(arr: PyObject) -> Self {
        nox::SpatialForce::from_torque(Vector::from_op(Noxpr::jax(arr))).into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };
        Ok(nox::SpatialForce::from_op(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialForce::from_op(Noxpr::jax(jax)).into()
    }

    fn force(&self) -> Result<PyObject, Error> {
        Ok(self.inner.force().into_op().to_jax()?)
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.force().into_op().to_jax()?)
    }

    fn torque(&self) -> Result<PyObject, Error> {
        Ok(self.inner.torque().into_op().to_jax()?)
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [6]).into())
    }

    fn __add__(&self, other: &SpatialForce) -> Self {
        (self.inner.clone() + other.inner.clone()).into()
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Quaternion {
    inner: nox::Quaternion<f64>,
}

impl From<nox::Quaternion<f64>> for Quaternion {
    fn from(inner: nox::Quaternion<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Quaternion {
    #[new]
    fn new(arr: PyObject) -> Self {
        Quaternion {
            inner: nox::Quaternion::from_op(Noxpr::jax(arr)),
        }
    }

    fn vector(&self) -> PyObject {
        self.inner.clone().into_op().to_jax().unwrap()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::Quaternion::from_op(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::Quaternion::from_op(Noxpr::jax(jax)).into()
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [4]).into())
    }

    #[staticmethod]
    fn from_axis_angle(axis: PyObject, angle: PyObject) -> Self {
        nox::Quaternion::from_axis_angle(
            Vector::from_op(Noxpr::jax(axis)),
            Scalar::from_op(Noxpr::jax(angle)),
        )
        .into()
    }

    fn normalize(&self) -> Self {
        self.inner.clone().normalize().into()
    }

    #[staticmethod]
    fn identity() -> Self {
        nox::Quaternion::identity().into()
    }

    pub fn __mul__(&self, rhs: Self) -> Self {
        let quat: nox::Quaternion<f64> = self.inner.clone().mul(rhs.inner);
        Self { inner: quat }
        //self.inner.clone().mul(rhs.inner).into()
    }

    pub fn __add__(&self, rhs: PyObject) -> Self {
        Python::with_gil(|py| {
            if let Ok(rhs) = rhs.extract::<Quaternion>(py) {
                self.inner.clone().add(rhs.inner).into()
            } else {
                dbg!(rhs
                    .getattr(py, "shape")
                    .unwrap()
                    .call_method0(py, "__repr__")
                    .unwrap()
                    .extract::<String>(py)
                    .unwrap());
                todo!()
            }
        })
    }

    pub fn __matmul__(&self, rhs: PyObject) -> PyObject {
        let vec = Vector::from_op(Noxpr::jax(rhs));
        self.inner.clone().mul(vec).into_op().to_jax().unwrap()
    }

    pub fn inverse(&self) -> Self {
        self.inner.clone().inverse().into()
    }
}

#[pyclass]
pub struct SpatialInertia {
    inner: nox::SpatialInertia<f64>,
}

impl From<nox::SpatialInertia<f64>> for SpatialInertia {
    fn from(inner: nox::SpatialInertia<f64>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl SpatialInertia {
    #[new]
    fn new(mass: PyObject, inertia: PyObject) -> Self {
        nox::SpatialInertia::new(
            Vector::<f64, 3>::from_op(Noxpr::jax(inertia)),
            Vector::<f64, 3>::zeros(),
            Scalar::<f64>::from_op(Noxpr::jax(mass)),
        )
        .into()
    }

    #[staticmethod]
    fn from_mass(arr: PyObject) -> Self {
        nox::SpatialInertia::from_mass(Scalar::from_op(Noxpr::jax(arr))).into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialInertia::from_op(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialInertia::from_op(Noxpr::jax(jax)).into()
    }

    fn mass(&self) -> Result<PyObject, Error> {
        Ok(self.inner.mass().into_op().to_jax()?)
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [7]).into())
    }
}
