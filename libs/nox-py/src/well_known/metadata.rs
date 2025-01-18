use crate::*;

use impeller2::component::Asset;
use pyo3::{intern, types::PySequence};

#[pyclass]
#[derive(Clone)]
pub struct EntityMetadata {
    name: String,
    color: Color,
}

#[pymethods]
impl EntityMetadata {
    #[new]
    pub fn new(name: String, color: Option<Color>) -> Self {
        let color = color.unwrap_or(Color::new(1.0, 1.0, 1.0));
        Self { name, color }
    }
}
