use crate::*;

use conduit::ComponentId;
use numpy::{ndarray::ArrayViewD, PyArray, PyUntypedArray};

use pyo3_polars::PyDataFrame;

#[pyclass]
pub struct Exec {
    pub exec: nox_ecs::WorldExec,
}

#[pymethods]
impl Exec {
    pub fn run(&mut self, client: &Client) -> Result<(), Error> {
        Python::with_gil(|_| self.exec.run(&client.client).map_err(Error::from))
    }

    pub fn history(&self) -> Result<PyDataFrame, Error> {
        let polars_world = self.exec.history.compact_to_world()?;
        let df = polars_world.join_archetypes()?;
        Ok(PyDataFrame(df))
    }

    fn column_array(
        this_cell: &PyCell<Self>,
        name: String,
    ) -> Result<&'_ numpy::PyUntypedArray, Error> {
        let mut this = this_cell.borrow_mut();
        let column = this.exec.column(ComponentId::new(&name))?;
        let dyn_array = column
            .column
            .dyn_ndarray()
            .ok_or(nox_ecs::Error::ComponentNotFound)?;
        fn untyped_pyarray<'py, T: numpy::Element + 'static>(
            view: &ArrayViewD<'_, T>,
            container: &'py PyAny,
        ) -> &'py PyUntypedArray {
            // # Safety
            // This is one of those things that I'm like 75% sure is safe enough,
            // but also close to 100% sure it breaks Rust's rules.
            // There are essentially two safety guarantees that we want to keep
            // when doing weird borrow stuff, ensure you aren't creating a reference
            // to free-ed / uninitialized/unaligned memory and to ensure that you are not
            // accidentally creating aliasing. We know we aren't doing the first one here,
            // because `Exec` is guaranteed to stay around as long as the `PyCell` is around.
            // We are technically breaking the 2nd rule, BUT, I think the way we are doing it is also ok.
            // What can happen is that we call `column_array`, then we call `run`, then we call `column_array` again.
            // We still have an outstanding reference to the array, and calling `column_array` again will cause the contents to be overwritten.
            // In most languages, this would cause all sorts of problems, but thankfully, in Python, we have the GIL to save the day.
            // While `exec` is run, the GIL is taken, so no one can access the old array result.
            // We never re-alloc the underlying buffer because lengths are immutable during execution.
            unsafe {
                let arr = PyArray::borrow_from_array(view, container);
                arr.as_untyped()
            }
        }
        match dyn_array {
            nox_ecs::DynArrayView::F64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::F32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U16(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::U8(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I64(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I32(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I16(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::I8(f) => Ok(untyped_pyarray(&f, this_cell)),
            nox_ecs::DynArrayView::Bool(f) => Ok(untyped_pyarray(&f, this_cell)),
        }
    }
}
