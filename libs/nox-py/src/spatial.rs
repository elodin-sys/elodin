use core::ops::{Add, Mul};

use nox::{self, Noxpr, Op, ReprMonad, Scalar, Tensor, Vector};
use pyo3::IntoPyObjectExt;
use pyo3::{prelude::*, types::PyTuple};

use crate::{Error, PyComponent};

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
    #[pyo3(signature = (arr=None, angular=None, linear=None))]
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
    fn metadata() -> PyComponent {
        PyComponent::from_component::<nox::SpatialTransform<f64>>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
    }

    fn __add__(&self, py: Python<'_>, rhs: PyObject) -> PyResult<PyObject> {
        if let Ok(s) = rhs.extract::<SpatialTransform>(py) {
            let op = self.inner.clone().add(s.inner).into_inner();
            let spatial_transform = SpatialTransform::from(nox::SpatialTransform::from_inner(op));
            Ok(spatial_transform.into_py_any(py)?)
        } else if let Ok(s) = rhs.extract::<SpatialMotion>(py) {
            let op = self.inner.clone().add(s.inner).into_inner();
            let spatial_motion = SpatialMotion::from(nox::SpatialMotion::from_inner(op));
            Ok(spatial_motion.into_py_any(py)?)
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
    #[pyo3(signature = (angular=None, linear=None))]
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
    fn metadata() -> PyComponent {
        PyComponent::from_component::<nox::SpatialMotion<f64>>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
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
    #[pyo3(signature = (arr=None, torque=None, linear=None))]
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
    fn metadata() -> PyComponent {
        PyComponent::from_component::<nox::SpatialForce<f64>>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
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
    fn metadata() -> PyComponent {
        PyComponent::from_component::<nox::Quaternion<f64>>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
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
            Ok(spatial_transform.into_py_any(py)?)
        } else if let Ok(s) = rhs.extract::<SpatialMotion>(py) {
            let op = self.inner.clone().mul(s.inner).into_inner();
            let spatial_motion = SpatialMotion::from(nox::SpatialMotion::from_inner(op));
            Ok(spatial_motion.into_py_any(py)?)
        } else if let Ok(s) = rhs.extract::<SpatialForce>(py) {
            let op = self.inner.clone().mul(s.inner).into_inner();
            let spatial_force = SpatialForce::from(nox::SpatialForce::from_inner(op));
            Ok(spatial_force.into_py_any(py)?)
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

fn validate_f64_array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    expected_shape: &[usize],
    label: &str,
) -> PyResult<bool> {
    if !obj.hasattr("shape")? || !obj.hasattr("dtype")? {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{label} must be a float64 array of shape {expected_shape:?}, got {}",
            obj.get_type().name()?
        )));
    }
    let shape: Vec<usize> = obj.getattr("shape")?.extract()?;
    if shape.as_slice() != expected_shape {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must have shape {expected_shape:?}, got shape {shape:?}"
        )));
    }
    let dtype_name: String = obj.getattr("dtype")?.getattr("name")?.extract()?;
    if dtype_name != "float64" {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{label} must have dtype float64, got {dtype_name}"
        )));
    }
    let tracer_ty = py.import("jax.core")?.getattr("Tracer")?;
    Ok(!obj.is_instance(&tracer_ty)?)
}

fn validate_mass_arg(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let m = if obj.is_instance_of::<pyo3::types::PyFloat>() {
        obj.extract::<f64>()?
    } else if obj.hasattr("shape")? && !obj.is_instance_of::<pyo3::types::PyBool>() {
        if !validate_f64_array(py, obj, &[], "SpatialInertia mass")? {
            return Ok(None);
        }
        obj.extract::<f64>()?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "SpatialInertia mass must be a float or 0-D float64 array, got {}",
            obj.get_type().name()?
        )));
    };
    if !m.is_finite() || m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SpatialInertia mass must be finite and > 0, got {m}"
        )));
    }
    Ok(Some(m))
}

fn validate_inertia_vec3_arg(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Option<[f64; 3]>> {
    let label = format!("SpatialInertia {name}");
    if !validate_f64_array(py, obj, &[3], &label)? {
        return Ok(None);
    }
    let vals: [f64; 3] = obj.extract()?;
    if !vals.iter().all(|v| v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{label} must contain finite values, got {vals:?}"
        )));
    }
    Ok(Some(vals))
}

fn symmetric_3x3_eigenvalues(diag: [f64; 3], off_diag: [f64; 3]) -> [f64; 3] {
    let [ixx, iyy, izz] = diag;
    let [ixy, ixz, iyz] = off_diag;
    let p1 = ixy * ixy + ixz * ixz + iyz * iyz;
    if p1 == 0.0 {
        let mut eigs = [ixx, iyy, izz];
        eigs.sort_by(f64::total_cmp);
        return eigs;
    }

    let q = (ixx + iyy + izz) / 3.0;
    let [b_xx, b_yy, b_zz] = [ixx - q, iyy - q, izz - q];
    let p = ((b_xx * b_xx + b_yy * b_yy + b_zz * b_zz + 2.0 * p1) / 6.0).sqrt();
    if p == 0.0 {
        return [q, q, q];
    }

    let det_b = b_xx * (b_yy * b_zz - iyz * iyz)
        - ixy * (ixy * b_zz - ixz * iyz)
        + ixz * (ixy * iyz - b_yy * ixz);
    let phi = (det_b / (2.0 * p * p * p)).clamp(-1.0, 1.0).acos() / 3.0;

    let eig3 = q + 2.0 * p * phi.cos();
    let eig1 = q + 2.0 * p * (phi + 2.0 * core::f64::consts::FRAC_PI_3).cos();
    let eig2 = 3.0 * q - eig1 - eig3;
    let mut eigs = [eig1, eig2, eig3];
    eigs.sort_by(f64::total_cmp);
    eigs
}

fn validate_inertia_tensor(diag: [f64; 3], off_diag: [f64; 3]) -> PyResult<()> {
    if !diag.iter().all(|v| v.is_finite()) || !off_diag.iter().all(|v| v.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SpatialInertia inertia tensor components must be finite",
        ));
    }
    let [ixx, iyy, izz] = diag;
    let [ixy, ixz, iyz] = off_diag;
    let det = ixx * (iyy * izz - iyz * iyz)
        + ixy * (ixz * iyz - ixy * izz)
        + ixz * (ixy * iyz - ixz * iyy);
    let [l1, l2, l3] = symmetric_3x3_eigenvalues(diag, off_diag);

    if ixx <= 0.0 || ixx * iyy - ixy * ixy <= 0.0 || det <= 0.0 || l1 <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SpatialInertia inertia tensor must be positive-definite (principal moments: [{l1}, {l2}, {l3}], det: {det})"
        )));
    }
    if l1 + l2 + 1e-12 * l3 < l3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "SpatialInertia inertia tensor violates the triangle inequality: principal moments [{l1}, {l2}, {l3}] do not satisfy l1 + l2 >= l3"
        )));
    }
    Ok(())
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
    #[pyo3(signature = (mass, inertia=None, inertia_off_diag=None))]
    fn new(
        py: Python<'_>,
        mass: PyObject,
        inertia: Option<PyObject>,
        inertia_off_diag: Option<PyObject>,
    ) -> PyResult<Self> {
        let mass_val = validate_mass_arg(py, mass.bind(py))?;
        let diag_val = match &inertia {
            Some(inertia_obj) => validate_inertia_vec3_arg(py, inertia_obj.bind(py), "inertia")?,
            None => mass_val.map(|m| [m, m, m]),
        };
        let off_diag_val = match &inertia_off_diag {
            Some(off_diag_obj) => {
                validate_inertia_vec3_arg(py, off_diag_obj.bind(py), "inertia_off_diag")?
            }
            None => Some([0.0, 0.0, 0.0]),
        };
        if let (Some(diag), Some(off_diag)) = (diag_val, off_diag_val) {
            validate_inertia_tensor(diag, off_diag)?;
        }

        let mass = Scalar::<f64>::from_inner(Noxpr::jax(mass));
        let inertia_off_diag = if let Some(off_diag) = inertia_off_diag {
            Vector::<f64, 3>::from_inner(Noxpr::jax(off_diag))
        } else {
            Vector::<f64, 3>::zeros()
        };
        let inertia = if let Some(inertia) = inertia {
            Vector::<f64, 3>::from_inner(Noxpr::jax(inertia))
        } else {
            Vector::<f64, 3>::ones() * mass.clone()
        };
        Ok(nox::SpatialInertia::new(inertia, inertia_off_diag, mass).into())
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
    fn from_array(py: Python<'_>, jax: PyObject) -> PyResult<Self> {
        let obj = jax.bind(py);
        if validate_f64_array(py, obj, &[7], "SpatialInertia.from_array")? {
            let vals: [f64; 7] = obj.extract()?;
            let mass = vals[6];
            if !mass.is_finite() || mass <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "SpatialInertia mass must be finite and > 0, got {mass}"
                )));
            }
            validate_inertia_tensor([vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]])?;
        }
        Ok(nox::SpatialInertia::from_inner(Noxpr::jax(jax)).into())
    }

    fn mass(&self) -> Result<PyObject, Error> {
        Ok(self.inner.mass().into_inner().to_jax()?)
    }

    fn inertia_diag(&self) -> Result<PyObject, Error> {
        Ok(self.inner.inertia_diag().into_inner().to_jax()?)
    }

    fn inertia_off_diag(&self) -> Result<PyObject, Error> {
        Ok(self.inner.inertia_off_diag().into_inner().to_jax()?)
    }

    fn inertia_matrix(&self) -> Result<PyObject, Error> {
        Ok(self.inner.inertia_matrix().into_inner().to_jax()?)
    }

    fn asarray(&self) -> Result<PyObject, Error> {
        Ok(self.inner.clone().into_inner().to_jax()?)
    }

    #[classattr]
    fn metadata() -> PyComponent {
        PyComponent::from_component::<nox::SpatialInertia<f64>>()
    }

    #[classattr]
    fn __metadata__() -> (PyComponent,) {
        (Self::metadata(),)
    }
}
