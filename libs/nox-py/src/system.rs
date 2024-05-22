use crate::*;

use nox_ecs::{IntoSystem, System};
use std::sync::Arc;

#[derive(Clone)]
pub struct PySystem {
    pub sys: PyObject,
}

impl System<nox_ecs::PipelineBuilder> for PySystem {
    type Arg = ();

    type Ret = ();

    fn init_builder(
        &self,
        in_builder: &mut nox_ecs::PipelineBuilder,
    ) -> Result<(), nox_ecs::Error> {
        let builder = std::mem::take(in_builder);
        let builder = PipelineBuilder { builder };
        *in_builder = Python::with_gil(move |py| {
            let builder = Bound::new(py, builder)?;
            self.sys.call_method1(py, "init", (builder.borrow_mut(),))?;
            let builder = std::mem::take(&mut builder.borrow_mut().builder);
            Ok::<_, nox_ecs::Error>(builder)
        })?;
        Ok(())
    }

    fn add_to_builder(
        &self,
        in_builder: &mut nox_ecs::PipelineBuilder,
    ) -> Result<(), nox_ecs::Error> {
        let builder = std::mem::take(in_builder);
        let builder = PipelineBuilder { builder };
        *in_builder = Python::with_gil(move |py| {
            let builder = Bound::new(py, builder)?;
            self.sys.call_method1(py, "call", (builder.borrow_mut(),))?;
            let builder = std::mem::take(&mut builder.borrow_mut().builder);
            Ok::<_, nox_ecs::Error>(builder)
        })?;
        Ok(())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct RustSystem {
    pub inner: Arc<dyn System<nox_ecs::PipelineBuilder, Arg = (), Ret = ()> + Send + Sync>,
}

#[pymethods]
impl RustSystem {
    fn init(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.inner.init_builder(&mut builder.builder)?;
        Ok(())
    }
    fn call(&self, builder: &mut PipelineBuilder) -> Result<(), Error> {
        self.inner.add_to_builder(&mut builder.builder)?;
        Ok(())
    }
    fn pipe(&self, sys: PyObject) -> RustSystem {
        let pipe = nox_ecs::Pipe::new(self.clone().into_system(), PySystem { sys });
        RustSystem {
            inner: Arc::new(ErasedSystem::new(pipe)),
        }
    }
    fn __or__(&self, sys: PyObject) -> RustSystem {
        self.pipe(sys)
    }
}

impl System<nox_ecs::PipelineBuilder> for RustSystem {
    type Arg = ();

    type Ret = ();

    fn init_builder(&self, builder: &mut nox_ecs::PipelineBuilder) -> Result<(), nox_ecs::Error> {
        self.inner.init_builder(builder)
    }

    fn add_to_builder(&self, builder: &mut nox_ecs::PipelineBuilder) -> Result<(), nox_ecs::Error> {
        self.inner.add_to_builder(builder)
    }
}
