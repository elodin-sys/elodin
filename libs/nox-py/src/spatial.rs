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

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_op().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        nox::SpatialTransform::from_op(Noxpr::jax(jax)).into()
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
    fn new(arr: PyObject) -> Self {
        nox::SpatialMotion::from_op(Noxpr::jax(arr)).into()
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
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        nox::SpatialMotion::from_op(Noxpr::jax(jax)).into()
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
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        nox::SpatialForce::from_op(Noxpr::jax(jax)).into()
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialForce::from_op(Noxpr::jax(jax)).into()
    }

    fn force(&self) -> Result<PyObject, Error> {
        Ok(self.inner.force().into_op().to_jax()?)
    }

    fn torque(&self) -> Result<PyObject, Error> {
        Ok(self.inner.torque().into_op().to_jax()?)
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [6]).into())
    }
}

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
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        nox::Quaternion::from_op(Noxpr::jax(jax)).into()
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::Quaternion::from_op(Noxpr::jax(jax)).into()
    }

    #[getter]
    fn shape(&self) -> PyObject {
        Python::with_gil(|py| PyTuple::new(py, [4]).into())
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
    fn unflatten(_aux: PyObject, jax: PyObject) -> Self {
        nox::SpatialInertia::from_op(Noxpr::jax(jax)).into()
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
