#![allow(clippy::await_holding_refcell_ref)]

use impeller2::types::{ComponentId, EntityId, LenPacket, PrimType};
use impeller2::vtable::builder::{pair, raw_field, schema, vtable};
use impeller2_stellar::Client;
use impeller2_wkt::{SetComponentMetadata, SetEntityMetadata, VTableMsg};
use numpy::{PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods, dtype};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::rc::Rc;
use stellarator::ExecutorHandle;

#[pyclass(unsendable)]
pub struct ElodinDB {
    client: Rc<RefCell<Client>>,
    #[pyo3(get)]
    addr: String,
    handle: ExecutorHandle,
}

#[pymethods]
impl ElodinDB {
    #[staticmethod]
    #[pyo3(signature = (addr = "[::]:0", path = None))]
    fn start(addr: &str, path: Option<PathBuf>) -> PyResult<Self> {
        let path = if let Some(path) = path {
            path
        } else {
            let tmp_dir =
                tempfile::tempdir().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            tmp_dir.keep()
        };
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address: {}", e)))?;

        let listener = stellarator::net::TcpListener::bind(addr)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to bind listener: {}", e)))?;
        let local_addr = listener
            .local_addr()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to get local address: {}", e)))?;
        let server = ::elodin_db::Server::from_listener(listener, path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start server: {}", e)))?;

        stellarator::struc_con::stellar(move || server.run());

        let handle = stellarator::Executor::enter();
        let client = handle
            .block_on(move || async move { Client::connect(local_addr).await })
            .map_err(|e| PyRuntimeError::new_err(format!("failed to connect: {}", e)))?;

        Ok(ElodinDB {
            client: Rc::new(RefCell::new(client)),
            addr: local_addr.to_string(),
            handle,
        })
    }

    #[staticmethod]
    fn connect(addr: &str) -> PyResult<Self> {
        let addr: SocketAddr = addr
            .parse()
            .map_err(|e| PyRuntimeError::new_err(format!("invalid address: {}", e)))?;

        let handle = stellarator::Executor::enter();
        let client = handle
            .block_on(move || Client::connect(addr))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to connect: {}", e)))?;

        let addr = addr.to_string();
        Ok(ElodinDB {
            client: Rc::new(RefCell::new(client)),
            addr,
            handle,
        })
    }

    fn send_table(
        &self,
        entity_id: u64,
        component_id: &str,
        data: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        let entity_id = EntityId(entity_id);
        let component_id = ComponentId::new(component_id);

        let packet_id = fastrand::u16(..).to_le_bytes();

        let shape: SmallVec<[u64; 4]> = data.shape().iter().map(|&x| x as u64).collect();

        let dtype = data.dtype();
        let elem_size = dtype.itemsize();
        let prim_type = dtype_to_prim_type(data.py(), dtype)?;

        let buf = unsafe { data.buf(elem_size) };

        let vtable_msg = VTableMsg {
            id: packet_id,
            vtable: vtable([raw_field(
                0,
                buf.len() as u16,
                schema(prim_type, &shape, pair(entity_id.0, component_id)),
            )]),
        };

        let mut table_packet = LenPacket::table(packet_id, buf.len());
        table_packet.extend_aligned(buf);

        let client = self.client.clone();
        #[allow(clippy::await_holding_refcell_ref)]
        self.handle.block_on(move || async move {
            let mut client = client.borrow_mut();
            client
                .send(&vtable_msg)
                .await
                .0
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to send vtable: {}", e)))?;

            client
                .send(table_packet)
                .await
                .0
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to send table: {}", e)))?;

            Ok::<(), PyErr>(())
        })
    }

    /// Set metadata for a component.
    ///
    /// Args:
    ///     component_id: The component identifier
    ///     name: Human-readable name for the component
    ///     metadata: Optional dictionary of metadata key-value pairs
    ///     asset: Optional flag indicating if this component is an asset
    ///
    /// Example:
    ///     ```python
    ///     db.set_component_metadata("position", "position",
    ///                               metadata={"element_names": "x,y,z"},
    ///                               asset=False)
    ///     ```
    #[pyo3(signature = (component_id, name = None, metadata=None, asset=None))]
    fn set_component_metadata(
        &self,
        component_id: &str,
        name: Option<&str>,
        metadata: Option<std::collections::HashMap<String, String>>,
        asset: Option<bool>,
    ) -> PyResult<()> {
        let name = name.unwrap_or(component_id);
        let component_id = ComponentId::new(component_id);
        let mut msg = SetComponentMetadata::new(component_id, name);

        if let Some(metadata) = metadata {
            msg = msg.metadata(metadata);
        }

        if let Some(asset) = asset {
            msg = msg.asset(asset);
        }

        let client = self.client.clone();
        self.handle.block_on(move || async move {
            let mut client = client.borrow_mut();
            client.send(&msg).await.0.map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set component metadata: {}", e))
            })?;
            Ok::<(), PyErr>(())
        })
    }

    /// Set metadata for an entity.
    ///
    /// Args:
    ///     entity_id: The numeric entity identifier
    ///     name: Human-readable name for the entity
    ///     metadata: Optional dictionary of metadata key-value pairs
    ///
    /// Example:
    ///     ```python
    ///     db.set_entity_metadata(42, "Satellite Alpha",
    ///                            metadata={"type": "LEO", "mission": "Earth observation"})
    ///     ```
    #[pyo3(signature = (entity_id, name, metadata=None))]
    fn set_entity_metadata(
        &self,
        entity_id: u64,
        name: &str,
        metadata: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<()> {
        let entity_id = EntityId(entity_id);
        let mut msg = SetEntityMetadata::new(entity_id, name);

        if let Some(metadata) = metadata {
            msg = msg.metadata(metadata);
        }

        let client = self.client.clone();
        self.handle.block_on(move || async move {
            let mut client = client.borrow_mut();
            client.send(&msg).await.0.map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to set entity metadata: {}", e))
            })?;
            Ok::<(), PyErr>(())
        })
    }
}

fn dtype_to_prim_type(py: Python, d: Bound<'_, PyArrayDescr>) -> PyResult<PrimType> {
    if d.is_equiv_to(&dtype::<f32>(py)) {
        Ok(PrimType::F32)
    } else if d.is_equiv_to(&dtype::<f64>(py)) {
        Ok(PrimType::F64)
    } else if d.is_equiv_to(&dtype::<i8>(py)) {
        Ok(PrimType::I8)
    } else if d.is_equiv_to(&dtype::<i16>(py)) {
        Ok(PrimType::I16)
    } else if d.is_equiv_to(&dtype::<i32>(py)) {
        Ok(PrimType::I32)
    } else if d.is_equiv_to(&dtype::<i64>(py)) {
        Ok(PrimType::I64)
    } else if d.is_equiv_to(&dtype::<u8>(py)) {
        Ok(PrimType::U8)
    } else if d.is_equiv_to(&dtype::<u16>(py)) {
        Ok(PrimType::U16)
    } else if d.is_equiv_to(&dtype::<u32>(py)) {
        Ok(PrimType::U32)
    } else if d.is_equiv_to(&dtype::<u64>(py)) {
        Ok(PrimType::U64)
    } else if d.is_equiv_to(&dtype::<bool>(py)) {
        Ok(PrimType::Bool)
    } else {
        Err(PyRuntimeError::new_err("Unsupported dtype"))
    }
}

trait PyUntypedArrayExt {
    unsafe fn buf(&self, elem_size: usize) -> &[u8];
}

impl PyUntypedArrayExt for Bound<'_, PyUntypedArray> {
    unsafe fn buf(&self, elem_size: usize) -> &[u8] {
        use numpy::PyUntypedArrayMethods;
        unsafe {
            if !self.is_c_contiguous() {
                panic!("array must be c-style contiguous")
            }
            let len = self.shape().iter().product::<usize>() * elem_size;
            let obj = &*self.as_array_ptr();
            std::slice::from_raw_parts(obj.data as *const u8, len)
        }
    }
}

#[pymodule]
fn elodin_db(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ElodinDB>()?;
    Ok(())
}
