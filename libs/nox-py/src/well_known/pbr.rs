use crate::*;

use nox_ecs::conduit;
use nox_ecs::conduit::Asset;

#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    pub inner: conduit::well_known::Mesh,
}

#[pymethods]
impl Mesh {
    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[staticmethod]
    pub fn cuboid(x: f32, y: f32, z: f32) -> Self {
        Self {
            inner: conduit::well_known::Mesh::cuboid(x, y, z),
        }
    }

    #[staticmethod]
    pub fn sphere(radius: f32) -> Self {
        Self {
            inner: conduit::well_known::Mesh::sphere(radius, 36, 18),
        }
    }

    pub fn asset_name(&self) -> &'static str {
        conduit::well_known::Mesh::ASSET_NAME
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Material {
    pub inner: conduit::well_known::Material,
}

#[pymethods]
impl Material {
    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    #[staticmethod]
    fn color(r: f32, g: f32, b: f32) -> Self {
        Material {
            inner: conduit::well_known::Material::color(r, g, b),
        }
    }
    pub fn asset_name(&self) -> &'static str {
        conduit::well_known::Material::ASSET_NAME
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Color {
    pub inner: conduit::well_known::Color,
}

#[pymethods]
impl Color {
    #[new]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color {
            inner: conduit::well_known::Color::rgb(r, g, b),
        }
    }

    #[classattr]
    pub const TURQUOISE: Self = Color {
        inner: conduit::well_known::Color::TURQUOISE,
    };

    #[classattr]
    pub const SLATE: Self = Color {
        inner: conduit::well_known::Color::SLATE,
    };

    #[classattr]
    pub const PUMPKIN: Self = Color {
        inner: conduit::well_known::Color::PUMPKIN,
    };

    #[classattr]
    pub const YOLK: Self = Color {
        inner: conduit::well_known::Color::YOLK,
    };

    #[classattr]
    pub const PEACH: Self = Color {
        inner: conduit::well_known::Color::PEACH,
    };

    #[classattr]
    pub const REDDISH: Self = Color {
        inner: conduit::well_known::Color::REDDISH,
    };

    #[classattr]
    pub const HYPERBLUE: Self = Color {
        inner: conduit::well_known::Color::HYPERBLUE,
    };

    #[classattr]
    pub const MINT: Self = Color {
        inner: conduit::well_known::Color::MINT,
    };
}

#[derive(Clone)]
#[pyclass]
pub struct Glb {
    pub inner: conduit::well_known::Glb,
}

#[pymethods]
impl Glb {
    #[new]
    pub fn new(url: String) -> Result<Self, Error> {
        let inner = conduit::well_known::Glb(url);
        Ok(Glb { inner })
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    pub fn asset_name(&self) -> &'static str {
        conduit::well_known::Glb::ASSET_NAME
    }
}
