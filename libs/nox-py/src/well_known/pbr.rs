use crate::*;

use impeller2::component::Asset;

#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    pub inner: impeller2_wkt::Mesh,
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
            inner: impeller2_wkt::Mesh::cuboid(x, y, z),
        }
    }

    #[staticmethod]
    pub fn sphere(radius: f32) -> Self {
        Self {
            inner: impeller2_wkt::Mesh::sphere(radius),
        }
    }

    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Mesh::NAME
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Material {
    pub inner: impeller2_wkt::Material,
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
            inner: impeller2_wkt::Material::color(r, g, b),
        }
    }
    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Material::NAME
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Color {
    pub inner: impeller2_wkt::Color,
}

#[pymethods]
impl Color {
    #[new]
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Color {
            inner: impeller2_wkt::Color::rgb(r, g, b),
        }
    }

    #[classattr]
    pub const TURQUOISE: Self = Color {
        inner: impeller2_wkt::Color::TURQUOISE,
    };

    #[classattr]
    pub const SLATE: Self = Color {
        inner: impeller2_wkt::Color::SLATE,
    };

    #[classattr]
    pub const PUMPKIN: Self = Color {
        inner: impeller2_wkt::Color::PUMPKIN,
    };

    #[classattr]
    pub const YOLK: Self = Color {
        inner: impeller2_wkt::Color::YOLK,
    };

    #[classattr]
    pub const PEACH: Self = Color {
        inner: impeller2_wkt::Color::PEACH,
    };

    #[classattr]
    pub const REDDISH: Self = Color {
        inner: impeller2_wkt::Color::REDDISH,
    };

    #[classattr]
    pub const HYPERBLUE: Self = Color {
        inner: impeller2_wkt::Color::HYPERBLUE,
    };

    #[classattr]
    pub const MINT: Self = Color {
        inner: impeller2_wkt::Color::MINT,
    };
}

#[derive(Clone)]
#[pyclass]
pub struct Glb {
    pub inner: impeller2_wkt::Glb,
}

#[pymethods]
impl Glb {
    #[new]
    pub fn new(url: String) -> Result<Self, Error> {
        let inner = impeller2_wkt::Glb(url);
        Ok(Glb { inner })
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }

    pub fn asset_name(&self) -> &'static str {
        impeller2_wkt::Glb::NAME
    }
}
