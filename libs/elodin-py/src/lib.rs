use std::sync::{Arc, OnceLock};

use bevy::prelude::{
    shape::{self, UVSphere},
    Color, Entity, World,
};
use elodin::{
    builder::{EntityBuilder, FixedJoint, Free, Revolute},
    spatial::{SpatialMotion, SpatialPos},
    Effect, Force, Inertia, Torque, XlaEffectors,
};
use elodin_conduit::{bevy::ComponentMap, ComponentId};
use nalgebra::{
    MatrixView3, MatrixView3x1, MatrixView4x1, Quaternion, UnitQuaternion, UnitVector3, Vector3,
};
use nox::{
    xla::{BufferArgsRef, PjRtLoadedExecutable},
    Client,
};
use numpy::{PyArrayLike1, PyArrayLike2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{PyBytes, PyTuple},
};

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
        effectors = None,
        trace_anchor = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        mass: Option<f64>,
        inertia: Option<PyArrayLike2<f64>>,
        mesh: Option<Mesh>,
        material: Option<Material>,
        joint: Option<Joint>,
        parent: Option<RigidBodyHandle>,
        body_pos: Option<PyArrayLike1<f64>>,
        effectors: Option<Vec<Effector>>,
        trace_anchor: Option<PyArrayLike1<f64>>,
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
        if let Some(trace_anchor) = trace_anchor {
            let Some(trace_anchor): Option<MatrixView3x1<f64>> = trace_anchor.try_as_matrix()
            else {
                return Err(PyErr::new::<PyTypeError, _>("pos must be a 1x3 matrix"));
            };

            inner = inner.trace(trace_anchor.into_owned());
        }

        if let Some(effectors) = effectors {
            let effectors: PyResult<Vec<_>> = effectors
                .into_iter()
                .map(|effector| {
                    let xla_comp = effector.to_xlo(py)?;
                    let compiled: OnceLock<PjRtLoadedExecutable> = OnceLock::new();
                    // NOTE: This code is ugly as hell and super hacky, but
                    // its ok since we will be deleting it all soon
                    Ok(Arc::new(
                        move |world: &mut World, entity: Entity, client: &nox::Client| {
                            let effect = {
                                let exec =
                                    compiled.get_or_init(|| client.0.compile(&xla_comp).unwrap());
                                let mut args = vec![];
                                let map = world.get_resource::<ComponentMap>().unwrap();
                                for component in &effector.components {
                                    if let Some(value) = map
                                        .0
                                        .get(&component.id)
                                        .and_then(|arg| arg.get(world, entity))
                                    {
                                        args.push(value.to_pjrt_buf(client).unwrap());
                                    }
                                }
                                let args = args.iter().collect::<BufferArgsRef>();
                                let buf =
                                    &exec.execute_buffers(args.untuple_result(true)).unwrap()[0];
                                let literal = buf.to_literal_sync().unwrap();
                                let out = literal.typed_buf::<f64>().unwrap();
                                Effect {
                                    force: Force(Vector3::new(out[0], out[1], out[2])),
                                    torque: Torque(Vector3::new(out[3], out[4], out[5])),
                                }
                            };
                            *world.get_mut::<Effect>(entity).unwrap() += effect;
                        },
                    )
                        as Arc<
                            dyn for<'a> Fn(&'a mut World, Entity, &'a Client) + Send + Sync,
                        >)
                })
                .collect();
            let effectors = effectors?;
            inner.xla_effectors = XlaEffectors(effectors);
        }
        Ok(RigidBody { inner })
    }
}

#[pyclass]
#[derive(Clone, Copy)]
pub enum ComponentType {
    // Primatives
    U8 = 0,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Bool,
    F32,
    F64,

    // Variable Size
    String,
    Bytes,

    // Tensors
    Vector3F32,
    Vector3F64,
    Matrix3x3F32,
    Matrix3x3F64,
    QuaternionF32,
    QuaternionF64,
    SpatialPosF32,
    SpatialPosF64,
    SpatialMotionF32,
    SpatialMotionF64,

    // Msgs
    Filter,
}

impl ComponentType {
    fn jax_zero<'py>(&self, py: Python<'py>) -> PyResult<&'py PyAny> {
        let jnp = PyModule::import(py, "numpy")?;
        let (shape, dtype) = match self {
            ComponentType::U8 => (vec![], jnp.getattr(intern!(py, "uint8"))?),
            ComponentType::U16 => (vec![], jnp.getattr(intern!(py, "uin16"))?),
            ComponentType::U32 => (vec![], jnp.getattr(intern!(py, "uint32"))?),
            ComponentType::U64 => (vec![], jnp.getattr(intern!(py, "uint64"))?),
            ComponentType::I8 => (vec![], jnp.getattr("int8")?),
            ComponentType::I16 => (vec![], jnp.getattr(intern!(py, "in16"))?),
            ComponentType::I32 => (vec![], jnp.getattr(intern!(py, "int32"))?),
            ComponentType::I64 => (vec![], jnp.getattr(intern!(py, "int64"))?),
            ComponentType::Bool => (vec![], jnp.getattr(intern!(py, "bool"))?),
            ComponentType::F32 => (vec![], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::F64 => (vec![], jnp.getattr(intern!(py, "float64"))?),
            ComponentType::Vector3F32 => (vec![3], jnp.getattr("float32")?),
            ComponentType::Vector3F64 => (vec![3], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::Matrix3x3F32 => (vec![3, 3], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::Matrix3x3F64 => (vec![3, 3], jnp.getattr(intern!(py, "float64"))?),
            ComponentType::QuaternionF32 => (vec![4], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::QuaternionF64 => (vec![4], jnp.getattr(intern!(py, "float64"))?),
            ComponentType::SpatialPosF32 => (vec![7], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::SpatialPosF64 => (vec![7], jnp.getattr(intern!(py, "float64"))?),
            ComponentType::SpatialMotionF32 => (vec![6], jnp.getattr(intern!(py, "float32"))?),
            ComponentType::SpatialMotionF64 => (vec![6], jnp.getattr(intern!(py, "float64"))?),
            ComponentType::String => todo!(),
            ComponentType::Bytes => todo!(),
            ComponentType::Filter => todo!(),
        };
        jnp.call_method1("zeros", (shape, dtype))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ComponentRef {
    id: ComponentId,
    ty: ComponentType,
}

#[pymethods]
impl ComponentRef {
    #[new]
    fn new(id: &str, ty: ComponentType) -> Self {
        ComponentRef {
            id: ComponentId::new(id),
            ty,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Effector {
    components: Vec<ComponentRef>,
    closure: PyObject,
}

#[pymethods]
impl Effector {
    #[new]
    fn new(components: Vec<ComponentRef>, closure: PyObject) -> Effector {
        Self {
            components,
            closure,
        }
    }
}

impl Effector {
    fn to_xlo(&self, py: Python<'_>) -> PyResult<nox::xla::XlaComputation> {
        let jax = PyModule::import(py, "jax")?;
        let args = self.components.iter().map(|r| r.ty.jax_zero(py).unwrap());
        let comp = jax
            .call_method1("xla_computation", (&self.closure,))?
            .call(PyTuple::new(py, args), None)?;
        let comp = comp.call_method0("as_serialized_hlo_module_proto")?;
        let comp = comp.downcast::<PyBytes>()?;
        let hlo_module = nox::xla::HloModuleProto::parse_binary(comp.as_bytes())
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(hlo_module.computation())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct SimBuilder(pub elodin::builder::SimBuilder);

#[pymethods]
impl SimBuilder {
    #[new]
    fn new() -> Self {
        SimBuilder(elodin::builder::SimBuilder::default())
    }

    fn body(&mut self, entity: RigidBody) -> RigidBodyHandle {
        RigidBodyHandle(self.0.entity(entity.inner))
    }

    fn zero_g(&mut self) {
        self.0.zero_g();
    }

    fn g_accel(&mut self, accel: PyArrayLike1<f64>) -> PyResult<()> {
        let Some(accel): Option<MatrixView3x1<f64>> = accel.try_as_matrix() else {
            return Err(PyErr::new::<PyTypeError, _>("gravity must be a 1x3 matrix"));
        };

        self.0.g_accel(SpatialMotion::linear(accel.into_owned()));
        Ok(())
    }

    fn gravity(&mut self, a: RigidBodyHandle, b: RigidBodyHandle) {
        self.0.gravity(a.0, b.0);
    }
}

#[derive(Clone)]
#[pyclass]
pub struct RigidBodyHandle(elodin::builder::RigidBodyHandle);

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

    #[staticmethod]
    pub fn sphere(radius: f32) -> Self {
        Mesh {
            mesh: bevy::prelude::Mesh::from(UVSphere {
                radius,
                ..Default::default()
            }),
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

    pub fn emissive(&mut self, r: f32, g: f32, b: f32) -> Self {
        self.material.emissive = Color::rgb_linear(r, g, b);
        self.clone()
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
        pos: Option<PyArrayLike1<f64>>,
        att: Option<PyArrayLike1<f64>>,
        vel: Option<PyArrayLike1<f64>>,
        ang_vel: Option<PyArrayLike1<f64>>,
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
        axis: PyArrayLike1<f64>,
        anchor: Option<PyArrayLike1<f64>>,
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
    elodin_editor::editor(move || builder.clone());
    Ok(())
}

#[pymodule]
pub fn elodin_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RigidBody>()?;
    m.add_class::<SimBuilder>()?;
    m.add_class::<Mesh>()?;
    m.add_class::<Material>()?;
    m.add_class::<Joint>()?;
    m.add_class::<Effector>()?;
    m.add_class::<ComponentRef>()?;
    m.add_class::<ComponentType>()?;
    m.add_function(wrap_pyfunction!(editor, m)?)?;
    Ok(())
}
