use core::ops::{Add, Mul};

use impeller::ComponentExt;
use nox_ecs::nox::{self, Noxpr, Op, ReprMonad, Scalar, Tensor, Vector};
use pyo3::{prelude::*, types::PyTuple};

use crate::{Component, Error, Metadata};

#[pyclass]
#[derive(Clone)]
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
    fn new(
        arr: Option<PyObject>,
        angular: Option<Quaternion>,
        linear: Option<PyObject>,
    ) -> PyResult<Self> {
        if let Some(arr) = arr {
            if linear.is_some() || angular.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both array and linear/angular",
                ));
            }
            Ok(nox::SpatialTransform::from_inner(Noxpr::jax(arr)).into())
        } else {
            let linear = linear
                .map(|arr| Tensor::<_, _, Op>::from_inner(Noxpr::jax(arr)))
                .unwrap_or_else(Vector::zeros);
            let angular = angular.unwrap_or_else(Quaternion::identity);
            Ok(Self {
                inner: nox::SpatialTransform::new(angular.inner, linear),
            })
        }
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }
    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast_bound::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialTransform::from_inner(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialTransform::from_inner(Noxpr::jax(jax)).into()
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.linear().into_inner().to_jax()?)
    }

    fn angular(&self) -> Quaternion {
        Quaternion {
            inner: self.inner.angular(),
        }
    }

    fn asarray(&self) -> Result<PyObject, Error> {
        Ok(self.inner.clone().into_inner().to_jax()?)
    }

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox::SpatialTransform::<f64>::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
    }

    fn __add__(&self, py: Python<'_>, rhs: PyObject) -> PyResult<PyObject> {
        if let Ok(s) = rhs.extract::<SpatialTransform>(py) {
            let op = self.inner.clone().add(s.inner).into_inner();
            let spatial_transform = SpatialTransform::from(nox::SpatialTransform::from_inner(op));
            Ok(spatial_transform.into_py(py).to_owned())
        } else if let Ok(s) = rhs.extract::<SpatialMotion>(py) {
            let op = self.inner.clone().add(s.inner).into_inner();
            let spatial_motion = SpatialMotion::from(nox::SpatialMotion::from_inner(op));
            Ok(spatial_motion.into_py(py).to_owned())
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported type for addition",
            ))
        }
    }
}

#[pyclass]
#[derive(Clone)]
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
    fn new(angular: Option<PyObject>, linear: Option<PyObject>) -> Self {
        let linear = linear
            .map(|arr| Tensor::<_, _, Op>::from_inner(Noxpr::jax(arr)))
            .unwrap_or_else(Vector::zeros);
        let angular = angular
            .map(|arr| Tensor::<_, _, Op>::from_inner(Noxpr::jax(arr)))
            .unwrap_or_else(Vector::zeros);
        nox::SpatialMotion::new(angular, linear).into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }
    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast_bound::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialMotion::from_inner(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialMotion::from_inner(Noxpr::jax(jax)).into()
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.linear().into_inner().to_jax()?)
    }

    fn angular(&self) -> Result<PyObject, Error> {
        Ok(self.inner.angular().into_inner().to_jax()?)
    }

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox::SpatialMotion::<f64>::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
    }

    fn __add__(&self, other: &SpatialMotion) -> Self {
        (self.inner.clone() + other.inner.clone()).into()
    }
}

#[pyclass]
#[derive(Clone)]
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
    fn new(
        arr: Option<PyObject>,
        torque: Option<PyObject>,
        linear: Option<PyObject>,
    ) -> PyResult<Self> {
        if let Some(arr) = arr {
            if linear.is_some() || torque.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Cannot specify both array and linear/torque",
                ));
            }
            Ok(nox::SpatialForce::from_inner(Noxpr::jax(arr)).into())
        } else {
            let linear = linear
                .map(|arr| Tensor::<_, _, Op>::from_inner(Noxpr::jax(arr)))
                .unwrap_or_else(Vector::zeros);
            let angular = torque
                .map(|arr| Tensor::<_, _, Op>::from_inner(Noxpr::jax(arr)))
                .unwrap_or_else(Vector::zeros);
            Ok(nox::SpatialForce::new(angular, linear).into())
        }
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast_bound::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };
        Ok(nox::SpatialForce::from_inner(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialForce::from_inner(Noxpr::jax(jax)).into()
    }

    fn force(&self) -> Result<PyObject, Error> {
        Ok(self.inner.force().into_inner().to_jax()?)
    }

    fn linear(&self) -> Result<PyObject, Error> {
        Ok(self.inner.force().into_inner().to_jax()?)
    }

    fn torque(&self) -> Result<PyObject, Error> {
        Ok(self.inner.torque().into_inner().to_jax()?)
    }

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox::SpatialForce::<f64>::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
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
            inner: nox::Quaternion::from_inner(Noxpr::jax(arr)),
        }
    }

    fn vector(&self) -> Result<PyObject, Error> {
        self.inner
            .clone()
            .into_inner()
            .to_jax()
            .map_err(Error::from)
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast_bound::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::Quaternion::from_inner(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::Quaternion::from_inner(Noxpr::jax(jax)).into()
    }

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox::Quaternion::<f64>::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
    }

    #[staticmethod]
    fn from_axis_angle(axis: PyObject, angle: PyObject) -> Self {
        nox::Quaternion::from_axis_angle(
            Tensor::<_, _, Op>::from_inner(Noxpr::jax(axis)),
            Tensor::<_, _, Op>::from_inner(Noxpr::jax(angle)),
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
    }

    pub fn __add__(&self, rhs: &Quaternion) -> Self {
        self.inner.clone().add(rhs.inner.clone()).into()
    }

    pub fn __matmul__(&self, py: Python<'_>, rhs: PyObject) -> Result<PyObject, Error> {
        if let Ok(s) = rhs.extract::<SpatialTransform>(py) {
            let op = self.inner.clone().mul(s.inner).into_inner();
            let spatial_transform = SpatialTransform::from(nox::SpatialTransform::from_inner(op));
            Ok(spatial_transform.into_py(py).to_owned())
        } else if let Ok(s) = rhs.extract::<SpatialMotion>(py) {
            let op = self.inner.clone().mul(s.inner).into_inner();
            let spatial_motion = SpatialMotion::from(nox::SpatialMotion::from_inner(op));
            Ok(spatial_motion.into_py(py).to_owned())
        } else if let Ok(s) = rhs.extract::<SpatialForce>(py) {
            let op = self.inner.clone().mul(s.inner).into_inner();
            let spatial_force = SpatialForce::from(nox::SpatialForce::from_inner(op));
            Ok(spatial_force.into_py(py).to_owned())
        } else {
            let vec = Vector::from_inner(Noxpr::jax(rhs));
            let op = self.inner.clone().mul(vec).into_inner();
            Ok(op.to_jax()?)
        }
    }

    pub fn inverse(&self) -> Self {
        self.inner.clone().inverse().into()
    }

    pub fn integrate_body(&self, arr: PyObject) -> Self {
        let body_delta = Vector::from_inner(Noxpr::jax(arr));
        self.inner.integrate_body(body_delta).into()
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
    fn new(mass: PyObject, inertia: Option<PyObject>) -> Self {
        let mass = Scalar::<f64>::from_inner(Noxpr::jax(mass));
        let momentum = Vector::<f64, 3>::zeros();
        let inertia = if let Some(inertia) = inertia {
            Vector::<f64, 3>::from_inner(Noxpr::jax(inertia))
        } else {
            Vector::<f64, 3>::ones() * mass.clone()
        };
        nox::SpatialInertia::new(inertia, momentum, mass).into()
    }

    fn flatten(&self) -> Result<((PyObject,), Option<()>), Error> {
        let jax = self.inner.clone().into_inner().to_jax()?;
        Ok(((jax,), None))
    }

    #[staticmethod]
    fn unflatten(py: Python<'_>, _aux: PyObject, jax: PyObject) -> pyo3::PyResult<Self> {
        let jax = if let Ok(tuple) = jax.downcast_bound::<PyTuple>(py) {
            tuple.get_item(0)?.into()
        } else {
            jax
        };

        Ok(nox::SpatialInertia::from_inner(Noxpr::jax(jax)).into())
    }

    #[staticmethod]
    fn from_array(jax: PyObject) -> Self {
        nox::SpatialInertia::from_inner(Noxpr::jax(jax)).into()
    }

    fn mass(&self) -> Result<PyObject, Error> {
        Ok(self.inner.mass().into_inner().to_jax()?)
    }

    fn inertia_diag(&self) -> Result<PyObject, Error> {
        Ok(self.inner.inertia_diag().into_inner().to_jax()?)
    }

    fn asarray(&self) -> Result<PyObject, Error> {
        Ok(self.inner.clone().into_inner().to_jax()?)
    }

    #[classattr]
    fn metadata() -> Metadata {
        Metadata {
            inner: nox::SpatialInertia::<f64>::metadata(),
        }
    }

    #[classattr]
    fn __metadata__() -> (Component,) {
        (Self::metadata().into(),)
    }
}
