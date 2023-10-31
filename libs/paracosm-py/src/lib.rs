use bevy::prelude::{shape, Color};
use nalgebra::{
    MatrixView3, MatrixView3x1, MatrixView4x1, Quaternion, UnitQuaternion, UnitVector3, Vector3,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use paracosm::{
    builder::{EntityBuilder, FixedJoint, Free, Revolute},
    spatial::{SpatialMotion, SpatialPos},
    Inertia,
};
use pyo3::{exceptions::PyTypeError, prelude::*};

#[derive(Clone)]
#[pyclass]
struct RigidBody {
    inner: EntityBuilder,
}

#[pymethods]
impl RigidBody {
    #[new]
    #[pyo3(signature = (
        mass = None,
        inertia = None,
        mesh = None,
        material = None,
        joint = None,
        parent = None,
        body_pos = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mass: Option<f64>,
        inertia: Option<PyReadonlyArray2<f64>>,
        mesh: Option<Mesh>,
        material: Option<Material>,
        joint: Option<Joint>,
        parent: Option<RigidBodyHandle>,
        body_pos: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let mut inner = EntityBuilder::default();
        if let Some(mass) = mass {
            inner = inner.mass(mass);
        }
        if let Some(inertia) = inertia {
            let Some(inertia): Option<MatrixView3<f64>> = inertia.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("inertia must be a 3x3 matrix"));
            };
            inner = inner.inertia(Inertia(inertia.into_owned()))
        }
        if let Some(joint) = joint {
            match joint.inner {
                JointInner::Free {
                    pos,
                    att,
                    vel,
                    ang_vel,
                } => {
                    inner = inner.joint(
                        Free::default()
                            .pos(SpatialPos::new(pos, att))
                            .vel(SpatialMotion::new(vel, ang_vel)),
                    );
                }
                JointInner::Revolute {
                    axis,
                    anchor,
                    pos,
                    vel,
                } => {
                    inner = inner.joint(Revolute::new(axis).anchor(anchor).pos(pos).vel(vel));
                }
                JointInner::Fixed => {
                    inner = inner.joint(FixedJoint);
                }
            }
        }
        if let Some(parent) = parent {
            inner = inner.parent(parent.0)
        }
        if let Some(mesh) = mesh {
            inner = inner.mesh(mesh.mesh);
        }
        if let Some(material) = material {
            inner = inner.material(material.material);
        }
        if let Some(body_pos) = body_pos {
            let Some(body_pos): Option<MatrixView3x1<f64>> = body_pos.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("pos must be a 1x3 matrix"));
            };

            inner = inner.body_pos(SpatialPos::linear(body_pos.into_owned()));
        }
        Ok(RigidBody { inner })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct SimBuilder(paracosm::builder::SimBuilder);

#[pymethods]
impl SimBuilder {
    #[new]
    fn new() -> Self {
        SimBuilder(paracosm::builder::SimBuilder::default())
    }

    fn body(&mut self, entity: RigidBody) -> RigidBodyHandle {
        RigidBodyHandle(self.0.entity(entity.inner))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct RigidBodyHandle(paracosm::builder::RigidBodyHandle);

#[derive(Clone)]
#[pyclass]
pub struct Mesh {
    mesh: bevy::prelude::Mesh,
}

#[pymethods]
impl Mesh {
    #[staticmethod]
    pub fn r#box(x_length: f32, y_length: f32, z_length: f32) -> Self {
        Mesh {
            mesh: bevy::prelude::Mesh::from(shape::Box::new(x_length, y_length, z_length)),
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Material {
    material: bevy::prelude::StandardMaterial,
}

#[pymethods]
impl Material {
    #[staticmethod]
    pub fn hex_color(hex: &str) -> PyResult<Self> {
        Ok(Material {
            material: Color::hex(hex)
                .map_err(|e| PyErr::new::<PyTypeError, _>(e.to_string()))?
                .into(),
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Joint {
    inner: JointInner,
}

#[derive(Clone)]
pub enum JointInner {
    Free {
        pos: Vector3<f64>,
        att: UnitQuaternion<f64>,
        vel: Vector3<f64>,
        ang_vel: Vector3<f64>,
    },

    Revolute {
        axis: UnitVector3<f64>,
        anchor: Vector3<f64>,
        pos: f64,
        vel: f64,
    },
    Fixed,
}

#[pymethods]
impl Joint {
    #[staticmethod]
    #[pyo3(signature = (pos = None, att = None, vel = None, ang_vel = None))]
    pub fn free(
        pos: Option<PyReadonlyArray1<f64>>,
        att: Option<PyReadonlyArray1<f64>>,
        vel: Option<PyReadonlyArray1<f64>>,
        ang_vel: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let pos = if let Some(pos) = pos {
            let Some(pos): Option<MatrixView3x1<f64>> = pos.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("pos must be a vector 3"));
            };
            pos.into_owned()
        } else {
            Vector3::zeros()
        };
        let att = if let Some(att) = att {
            let Some(att): Option<MatrixView4x1<f64>> = att.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("pos must be a quaternion"));
            };
            UnitQuaternion::new_normalize(Quaternion::from_vector(att.into_owned()))
        } else {
            UnitQuaternion::identity()
        };
        let vel = if let Some(vel) = vel {
            let Some(vel): Option<MatrixView3x1<f64>> = vel.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("vel must be a vector 3"));
            };
            vel.into_owned()
        } else {
            Vector3::zeros()
        };

        let ang_vel = if let Some(ang_vel) = ang_vel {
            let Some(ang_vel): Option<MatrixView3x1<f64>> = ang_vel.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("ang_vel must be a vector 3"));
            };
            ang_vel.into_owned()
        } else {
            Vector3::zeros()
        };

        Ok(Joint {
            inner: JointInner::Free {
                pos,
                att,
                vel,
                ang_vel,
            },
        })
    }

    #[staticmethod]
    #[pyo3(signature = (axis, anchor = None, pos = 0.0, vel = 0.0))]
    pub fn revolute(
        axis: PyReadonlyArray1<f64>,
        anchor: Option<PyReadonlyArray1<f64>>,
        pos: f64,
        vel: f64,
    ) -> PyResult<Self> {
        let Some(axis): Option<MatrixView3x1<f64>> = axis.try_as_matrix() else {
            return Err(PyErr::new::<PyTypeError, _>("axis must be a vector 3"));
        };
        let axis = UnitVector3::new_normalize(axis.into_owned());
        let anchor = if let Some(anchor) = anchor {
            let Some(anchor): Option<MatrixView3x1<f64>> = anchor.try_as_matrix() else {
                return Err(PyErr::new::<PyTypeError, _>("anchor must be a vector 3"));
            };
            anchor.into_owned()
        } else {
            Vector3::zeros()
        };

        Ok(Joint {
            inner: JointInner::Revolute {
                axis,
                anchor,
                pos,
                vel,
            },
        })
    }

    #[staticmethod]
    pub fn fixed() -> Self {
        Joint {
            inner: JointInner::Fixed,
        }
    }
}

#[pyfunction]
fn editor(py: Python<'_>, callable: PyObject) -> PyResult<()> {
    let builder = callable.call0(py)?;
    let builder: SimBuilder = builder.extract(py).unwrap();
    let builder = builder.0;
    paracosm_editor::editor(|| builder.clone());
    Ok(())
}

#[pymodule]
fn paracosm_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RigidBody>()?;
    m.add_class::<SimBuilder>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Material>()?;
    m.add_class::<Joint>()?;
    m.add_function(wrap_pyfunction!(editor, m)?)?;
    Ok(())
}
