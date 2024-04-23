use crate::*;

use std::{marker::PhantomData, ops::Deref};

use conduit::ComponentId;
use nox_ecs::{
    conduit,
    nox::{self, ArrayTy, Noxpr, NoxprNode, NoxprTy},
    ComponentArray,
};

use pyo3::types::PyTuple;

#[pyclass]
#[derive(Default)]
pub struct PipelineBuilder {
    pub builder: nox_ecs::PipelineBuilder,
}

#[pymethods]
impl PipelineBuilder {
    pub fn init_var(&mut self, name: String, ty: ComponentType) -> Result<(), Error> {
        let id = ComponentId::new(&name);
        if self.builder.param_ids.contains(&id) {
            return Ok(());
        }
        let column = self
            .builder
            .world
            .column_by_id(id)
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        let ty: conduit::ComponentType = ty.into();
        let len = column.column.len();
        let shape = std::iter::once(len as i64).chain(ty.shape).collect();
        let op = Noxpr::parameter(
            self.builder.param_ops.len() as i64,
            NoxprTy::ArrayTy(ArrayTy {
                element_type: ty.primitive_ty.element_type(),
                shape, // FIXME
            }),
            format!("{:?}::{}", id, self.builder.param_ops.len()),
        );
        self.builder.param_ops.push(op.clone());
        self.builder.param_ids.push(id);

        Ok(())
    }

    pub fn var_arrays(&mut self, py: Python<'_>) -> Result<Vec<PyObject>, Error> {
        let mut res = vec![];
        for p in &self.builder.param_ops {
            let NoxprNode::Param(p) = p.deref() else {
                continue;
            };
            let jnp = py.import("jax.numpy")?;
            let NoxprTy::ArrayTy(ty) = &p.ty else {
                unreachable!()
            };
            let dtype = nox::jax::dtype(&ty.element_type)?;
            let shape = PyTuple::new(py, ty.shape.iter().collect::<Vec<_>>());
            let arr = jnp.call_method1("zeros", (shape, dtype))?; // NOTE(sphw): this could be a huge bottleneck
            res.push(arr.into());
        }
        Ok(res)
    }

    pub fn inject_args(&mut self, args: Vec<PyObject>) -> Result<(), Error> {
        let builder = &mut self.builder;
        assert_eq!(args.len(), builder.param_ids.len());
        let nox_ecs::PipelineBuilder {
            vars,
            world,
            param_ids,
            ..
        } = builder;
        for (arg, id) in args.into_iter().zip(param_ids.iter()) {
            let column = world
                .column_by_id(*id)
                .ok_or(nox_ecs::Error::ComponentNotFound)?;
            let len = column.column.len();
            let array = ComponentArray {
                buffer: Noxpr::jax(arg),
                phantom_data: PhantomData,
                len,
                entity_map: column.entities.entity_map(),
            };
            vars.insert(*id, array.into());
        }
        Ok(())
    }

    pub fn ret_vars(&self, py: Python<'_>) -> Result<PyObject, Error> {
        let vars = self
            .builder
            .vars
            .values()
            .map(|var| {
                let var = var.borrow();
                var.buffer().to_jax()
            })
            .collect::<Result<Vec<_>, nox_ecs::nox::Error>>()?;
        Ok(PyTuple::new(py, vars).into())
    }
}
