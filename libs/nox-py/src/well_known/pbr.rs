use crate::*;

use std::path::PathBuf;

use nox_ecs::conduit;
use nox_ecs::conduit::Asset;

#[pyclass]
#[derive(Clone)]
pub struct Pbr {
    pub inner: conduit::well_known::Pbr,
}

#[pymethods]
impl Pbr {
    #[new]
    fn new(mesh: Mesh, material: Material) -> Self {
        Self {
            inner: conduit::well_known::Pbr::Bundle {
                mesh: mesh.inner,
                material: material.inner,
            },
        }
    }

    #[staticmethod]
    fn from_url(url: String) -> Result<Self, Error> {
        let inner = conduit::well_known::Pbr::Url(url);
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_path(path: PathBuf) -> Result<Self, Error> {
        let inner = conduit::well_known::Pbr::path(path)?;
        Ok(Self { inner })
    }

    pub fn asset_id(&self) -> u64 {
        self.inner.asset_id().0
    }

    pub fn bytes(&self) -> Result<PyBufBytes, Error> {
        let bytes = postcard::to_allocvec(&self.inner).unwrap().into();
        Ok(PyBufBytes { bytes })
    }
}

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
}
