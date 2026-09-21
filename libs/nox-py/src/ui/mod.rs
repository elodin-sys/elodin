//! `elodin.ui` — typed schematic builders that emit canonical KDL.
//!
//! Phase 1–2: builders over [`impeller2_wkt`]; EQL via strings or Python
//! `Expr` objects. Phase 3 adds watch/push + build-error metadata.

mod builders;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use impeller2::types::IntoLenPacket;
use impeller2_kdl::{
    apply_overlay as apply_overlay_model, extract_overlay as extract_overlay_model,
    overlay_asset_key, parse_overlay, parse_schematic, serialize_overlay, serialize_schematic,
};
use impeller2_wkt::{DISPLAY_KERNEL_ASSET_PREFIX, Schematic, SetDbConfig, StoreAsset};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

pub use builders::*;

const ACTIVE_SCHEMATIC_KEY: &str = "schematics/main.kdl";

/// Python-facing schematic: wraps [`Schematic`] and emits canonical KDL.
#[pyclass(name = "Schematic", module = "elodin.ui")]
#[derive(Clone, Debug)]
pub struct PySchematic {
    pub(crate) inner: Schematic,
    pub(crate) kernel_assets: HashMap<String, Vec<u8>>,
}

impl PySchematic {
    pub fn from_inner(inner: Schematic) -> Self {
        Self {
            inner,
            kernel_assets: HashMap::new(),
        }
    }

    pub fn from_parts(inner: Schematic, kernel_assets: HashMap<String, Vec<u8>>) -> Self {
        Self {
            inner,
            kernel_assets,
        }
    }

    pub fn emit_kdl_string(&self) -> String {
        serialize_schematic(&self.inner)
    }
}

#[pymethods]
impl PySchematic {
    /// Parse KDL text into a schematic (FR-11).
    #[staticmethod]
    fn from_kdl(text: &str) -> PyResult<Self> {
        let inner = parse_schematic(text).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            kernel_assets: HashMap::new(),
        })
    }

    /// Emit deterministic KDL (FR-2 / FR-3).
    fn emit_kdl(&self) -> String {
        self.emit_kdl_string()
    }

    /// Merge a layout overlay (KDL text) into this schematic (FR-9).
    fn apply_overlay(&mut self, overlay_kdl: &str) -> PyResult<()> {
        let overlay = parse_overlay(overlay_kdl)
            .map_err(|e| PyValueError::new_err(format!("parse overlay: {e}")))?;
        apply_overlay_model(&mut self.inner, &overlay);
        Ok(())
    }

    /// Emit a layout overlay for the current shares / window rects.
    fn extract_overlay(&self) -> String {
        serialize_overlay(&extract_overlay_model(&self.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "Schematic(elems={}, theme={}, timeline={}, frame={:?})",
            self.inner.elems.len(),
            self.inner.theme.is_some(),
            self.inner.timeline.is_some(),
            self.inner.frame,
        )
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other) = other.extract::<PyRef<'_, PySchematic>>() {
            Ok(self.inner == other.inner)
        } else {
            Ok(false)
        }
    }
}

/// Build a schematic from panels/objects plus optional globals.
#[pyfunction]
#[pyo3(signature = (
    *elems,
    coordinate=None,
    theme=None,
    timeline=None,
    skybox=None,
    environment=None,
    telemetry_mode=false,
))]
fn schematic(
    elems: Vec<Bound<'_, PyAny>>,
    coordinate: Option<Bound<'_, PyAny>>,
    theme: Option<Bound<'_, PyAny>>,
    timeline: Option<Bound<'_, PyAny>>,
    skybox: Option<String>,
    environment: Option<Bound<'_, PyAny>>,
    telemetry_mode: bool,
) -> PyResult<PySchematic> {
    let mut inner = Schematic {
        telemetry_mode,
        ..Schematic::default()
    };

    if let Some(coord) = coordinate {
        let c = builders::extract_coordinate(&coord)?;
        inner.frame = Some(c.frame);
        inner.origin = c.origin;
        inner.body = c.body;
    }
    if let Some(t) = theme {
        inner.theme = Some(builders::extract_theme(&t)?);
    }
    if let Some(t) = timeline {
        inner.timeline = Some(builders::extract_timeline(&t)?);
    }
    if let Some(name) = skybox {
        inner.skybox = Some(impeller2_wkt::SkyboxConfig { name });
    }
    if let Some(environment) = environment {
        inner.environment = Some(builders::extract_environment(&environment)?);
    }

    for elem in elems {
        push_elem(&mut inner, &elem)?;
    }

    Ok(PySchematic {
        inner,
        kernel_assets: builders::take_kernel_assets(),
    })
}

fn push_elem(schematic: &mut Schematic, obj: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(panel) = obj.extract::<PyRef<'_, PyPanel>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::Panel(panel.inner.clone()));
        return Ok(());
    }
    if let Ok(obj3d) = obj.extract::<PyRef<'_, PyObject3D>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::Object3d(obj3d.inner.clone()));
        return Ok(());
    }
    if let Ok(line) = obj.extract::<PyRef<'_, PyLine3d>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::Line3d(line.inner.clone()));
        return Ok(());
    }
    if let Ok(arrow) = obj.extract::<PyRef<'_, PyVectorArrow>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::VectorArrow(
                arrow.inner.clone(),
            ));
        return Ok(());
    }
    if let Ok(mesh) = obj.extract::<PyRef<'_, PyWorldMesh>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::WorldMesh(mesh.inner.clone()));
        return Ok(());
    }
    if let Ok(window) = obj.extract::<PyRef<'_, PyWindow>>() {
        schematic
            .elems
            .push(impeller2_wkt::SchematicElem::Window(window.inner.clone()));
        return Ok(());
    }
    Err(PyTypeError::new_err(
        "schematic elements must be Panel, Object3D, Line3d, VectorArrow, WorldMesh, or Window",
    ))
}

/// Write schematic KDL to a filesystem path (FR-5).
#[pyfunction]
fn write(schematic: &PySchematic, path: PathBuf) -> PyResult<()> {
    let kdl = schematic.emit_kdl_string();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| {
            PyRuntimeError::new_err(format!("create_dir_all {}: {e}", parent.display()))
        })?;
    }
    std::fs::write(&path, kdl)
        .map_err(|e| PyRuntimeError::new_err(format!("write {}: {e}", path.display())))?;
    if !schematic.kernel_assets.is_empty() {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        let kernel_dir = match parent {
            Some(parent) => parent.join("kernels"),
            None => PathBuf::from("kernels"),
        };
        std::fs::create_dir_all(&kernel_dir).map_err(|e| {
            PyRuntimeError::new_err(format!("create_dir_all {}: {e}", kernel_dir.display()))
        })?;
        for (key, bytes) in &schematic.kernel_assets {
            let hash = key
                .strip_prefix(DISPLAY_KERNEL_ASSET_PREFIX)
                .unwrap_or(key.rsplit('/').next().unwrap_or(key));
            let sidecar = kernel_dir.join(hash);
            std::fs::write(&sidecar, bytes).map_err(|e| {
                PyRuntimeError::new_err(format!("write {}: {e}", sidecar.display()))
            })?;
        }
    }
    Ok(())
}

/// Push schematic to a live DB: StoreAsset + set `schematic.active` (FR-5).
#[pyfunction]
#[pyo3(signature = (schematic, db, key = None))]
fn push(schematic: &PySchematic, db: &str, key: Option<String>) -> PyResult<()> {
    let key = key.unwrap_or_else(|| ACTIVE_SCHEMATIC_KEY.to_string());
    let addr = SocketAddr::from_str(db)
        .map_err(|e| PyValueError::new_err(format!("invalid db address {db:?}: {e}")))?;
    let bytes = schematic.emit_kdl_string().into_bytes();
    let store = StoreAsset {
        key: key.clone(),
        bytes,
    };
    let mut metadata = HashMap::new();
    metadata.insert("schematic.active".to_string(), key);
    // Clear any previous watch/build error on successful push.
    metadata.insert("ui.build_error".to_string(), String::new());
    let config = SetDbConfig {
        recording: None,
        metadata,
    };

    let kernel_assets = schematic.kernel_assets.clone();
    crate::db::block_on(move || async move {
        let mut client = impeller2_stellar::Client::connect(addr)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("connect to {addr}: {e}")))?;
        for (asset_key, bytes) in kernel_assets {
            let kernel_store = StoreAsset {
                key: asset_key,
                bytes,
            };
            client
                .send((&kernel_store).into_len_packet())
                .await
                .0
                .map_err(|e| PyRuntimeError::new_err(format!("StoreAsset kernel: {e}")))?;
        }
        client
            .send((&store).into_len_packet())
            .await
            .0
            .map_err(|e| PyRuntimeError::new_err(format!("StoreAsset: {e}")))?;
        client
            .send((&config).into_len_packet())
            .await
            .0
            .map_err(|e| PyRuntimeError::new_err(format!("SetDbConfig: {e}")))?;
        Ok::<(), PyErr>(())
    })?;
    Ok(())
}

/// Compile-check StableHLO text with the shared Cranelift backend.
#[pyfunction]
fn validate_stablehlo(mlir: &str) -> PyResult<()> {
    cranelift_mlir::display_kernel::DisplayKernelExec::validate(mlir).map_err(PyValueError::new_err)
}

const BUILD_ERROR_KEY: &str = "ui.build_error";

/// Publish or clear a schematic build error for the editor status bar (FR-8).
#[pyfunction]
#[pyo3(signature = (db, message=None))]
fn set_build_error(db: &str, message: Option<String>) -> PyResult<()> {
    let addr = SocketAddr::from_str(db)
        .map_err(|e| PyValueError::new_err(format!("invalid db address {db:?}: {e}")))?;
    let mut metadata = HashMap::new();
    // Empty string deletes the key (see DB apply_set_db_config).
    metadata.insert(BUILD_ERROR_KEY.to_string(), message.unwrap_or_default());
    let config = SetDbConfig {
        recording: None,
        metadata,
    };
    crate::db::block_on(move || async move {
        let mut client = impeller2_stellar::Client::connect(addr)
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("connect to {addr}: {e}")))?;
        client
            .send((&config).into_len_packet())
            .await
            .0
            .map_err(|e| PyRuntimeError::new_err(format!("SetDbConfig: {e}")))?;
        Ok::<(), PyErr>(())
    })?;
    Ok(())
}

/// Parse `default_content` for [`crate::WorldBuilder::schematic`]: `str` or [`PySchematic`].
///
/// Returns the KDL text plus any display-kernel sidecar bytes that must be
/// stored into the DB so the editor can fetch them.
pub fn extract_schematic_content(
    obj: &Bound<'_, PyAny>,
) -> PyResult<(String, HashMap<String, Vec<u8>>)> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok((s, HashMap::new()));
    }
    if let Ok(schematic) = obj.extract::<PyRef<'_, PySchematic>>() {
        return Ok((schematic.emit_kdl_string(), schematic.kernel_assets.clone()));
    }
    Err(PyTypeError::new_err(
        "schematic content must be str or elodin.ui.Schematic",
    ))
}

pub fn register(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child = PyModule::new(parent_module.py(), "ui")?;
    child.add_class::<PySchematic>()?;
    child.add_class::<PyPanel>()?;
    child.add_class::<PyObject3D>()?;
    child.add_class::<PyLine3d>()?;
    child.add_class::<PyVectorArrow>()?;
    child.add_class::<PyWorldMesh>()?;
    child.add_class::<PyWindow>()?;
    child.add_class::<PyMesh>()?;
    child.add_class::<PyJoint>()?;
    child.add_class::<PyCoordinate>()?;
    child.add_class::<PyTheme>()?;
    child.add_class::<PyTimeline>()?;
    child.add_class::<PyColor>()?;
    child.add_class::<PyEnvironment>()?;
    child.add_class::<PySun>()?;
    child.add_class::<PyAtmosphere>()?;
    child.add_class::<PyEarth>()?;
    child.add_class::<PyBloom>()?;
    child.add_class::<PyIcon>()?;
    child.add_class::<PyVisibilityRange>()?;
    child.add_class::<PyThruster>()?;
    child.add_class::<PyThrusterLight>()?;
    child.add_function(wrap_pyfunction!(schematic, &child)?)?;
    child.add_function(wrap_pyfunction!(from_kdl, &child)?)?;
    child.add_function(wrap_pyfunction!(to_python, &child)?)?;
    child.add_function(wrap_pyfunction!(write, &child)?)?;
    child.add_function(wrap_pyfunction!(push, &child)?)?;
    child.add_function(wrap_pyfunction!(set_build_error, &child)?)?;
    child.add_function(wrap_pyfunction!(validate_stablehlo, &child)?)?;
    child.add_function(wrap_pyfunction!(overlay_key, &child)?)?;
    child.add_function(wrap_pyfunction!(apply_overlay_kdl, &child)?)?;
    child.add_function(wrap_pyfunction!(extract_overlay_kdl, &child)?)?;
    builders::register_builders(&child)?;
    // Do not register as `sys.modules["elodin.ui"]` — that would shadow the
    // Python package (`elodin/ui/`) which wraps this native submodule and adds
    // Expr/Schema/watch. Native lives at `elodin.elodin.ui` (like monte_carlo).
    parent_module.add_submodule(&child)?;
    Ok(())
}

#[pyfunction]
fn from_kdl(text: &str) -> PyResult<PySchematic> {
    PySchematic::from_kdl(text)
}

#[pyfunction]
#[pyo3(signature = (text, source_name=None))]
fn to_python(text: &str, source_name: Option<&str>) -> PyResult<String> {
    impeller2_kdl::schematic_to_python(text, source_name)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
fn overlay_key(schematic_key: &str) -> String {
    overlay_asset_key(schematic_key)
}

/// `ui.apply_overlay(schematic, overlay_kdl) -> Schematic` (FR-9).
#[pyfunction]
#[pyo3(name = "apply_overlay")]
fn apply_overlay_kdl(schematic: &PySchematic, overlay_kdl: &str) -> PyResult<PySchematic> {
    let mut out = schematic.clone();
    out.apply_overlay(overlay_kdl)?;
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "extract_overlay")]
fn extract_overlay_kdl(schematic: &PySchematic) -> String {
    schematic.extract_overlay()
}
