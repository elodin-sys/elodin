use crate::buffer_view::BufferView;
use crate::element_type::ElementType;
use crate::error::{self, Result};
use crate::ffi;
use crate::session::Session;

#[derive(Debug, Clone)]
pub struct BufferSpec {
    pub byte_len: usize,
    pub shape: Vec<i64>,
    pub element_type: ElementType,
}

struct ArenaSlot {
    offset: usize,
    byte_len: usize,
    _subspan: DeviceBuffer,
    view: BufferView,
}

pub struct DeviceBuffer {
    pub(crate) ptr: *mut ffi::iree_hal_buffer_t,
}

impl DeviceBuffer {
    pub fn allocate(session: &Session, byte_len: usize) -> Result<Self> {
        let params = ffi::iree_hal_buffer_params_t {
            usage: ffi::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DEFAULT.0,
            access: 0,
            type_: ffi::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL.0
                | ffi::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_HOST_VISIBLE.0,
            queue_affinity: 0,
            min_alignment: 0,
        };
        let mut out: *mut ffi::iree_hal_buffer_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_allocator_allocate_buffer(
                session.device_allocator(),
                params,
                byte_len as ffi::iree_device_size_t,
                &mut out,
            )
        };
        error::check(status)?;
        Ok(Self { ptr: out })
    }

    pub fn byte_len(&self) -> usize {
        unsafe { ffi::iree_hal_buffer_byte_length(self.ptr) as usize }
    }

    pub fn upload(&self, session: &Session, source: &[u8], offset: usize) -> Result<()> {
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let status = unsafe {
            ffi::iree_hal_device_transfer_h2d(
                session.device(),
                source.as_ptr() as *const std::ffi::c_void,
                self.ptr,
                offset as ffi::iree_device_size_t,
                source.len() as ffi::iree_device_size_t,
                ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT.0,
                timeout,
            )
        };
        error::check(status)
    }

    pub fn download(&self, session: &Session, target: &mut [u8], offset: usize) -> Result<()> {
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let status = unsafe {
            ffi::iree_hal_device_transfer_d2h(
                session.device(),
                self.ptr,
                offset as ffi::iree_device_size_t,
                target.as_mut_ptr() as *mut std::ffi::c_void,
                target.len() as ffi::iree_device_size_t,
                ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT.0,
                timeout,
            )
        };
        error::check(status)
    }

    pub fn copy_from_buffer(
        &self,
        session: &Session,
        source: &DeviceBuffer,
        source_offset: usize,
        target_offset: usize,
        byte_len: usize,
    ) -> Result<()> {
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let status = unsafe {
            ffi::iree_hal_device_transfer_d2d(
                session.device(),
                source.ptr,
                source_offset as ffi::iree_device_size_t,
                self.ptr,
                target_offset as ffi::iree_device_size_t,
                byte_len as ffi::iree_device_size_t,
                ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT.0,
                timeout,
            )
        };
        error::check(status)
    }

    pub fn subspan(&self, offset: usize, byte_len: usize) -> Result<Self> {
        let mut out: *mut ffi::iree_hal_buffer_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_buffer_subspan(
                self.ptr,
                offset as ffi::iree_device_size_t,
                byte_len as ffi::iree_device_size_t,
                ffi::iree_allocator_system(),
                &mut out,
            )
        };
        error::check(status)?;
        Ok(Self { ptr: out })
    }

    pub fn create_view(
        &self,
        shape: &[i64],
        element_type: ElementType,
        encoding_type: u32,
    ) -> Result<BufferView> {
        let shape_dims: Vec<ffi::iree_hal_dim_t> =
            shape.iter().map(|&d| d as ffi::iree_hal_dim_t).collect();
        let mut out: *mut ffi::iree_hal_buffer_view_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_buffer_view_create(
                self.ptr,
                shape_dims.len(),
                shape_dims.as_ptr(),
                element_type.to_hal(),
                encoding_type,
                ffi::iree_allocator_system(),
                &mut out,
            )
        };
        error::check(status)?;
        Ok(BufferView { ptr: out })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_hal_buffer_release(self.ptr) };
        }
    }
}

pub struct DeviceArena {
    buffer: DeviceBuffer,
    slots: Vec<ArenaSlot>,
    upload_staging: Vec<u8>,
    download_staging: Vec<u8>,
}

impl DeviceArena {
    pub fn new(session: &Session, specs: &[BufferSpec]) -> Result<Self> {
        let mut total_len = 0usize;
        let mut offsets = Vec::with_capacity(specs.len());
        for spec in specs {
            let aligned = align_up(total_len, 16);
            offsets.push(aligned);
            total_len = aligned + spec.byte_len;
        }

        let buffer = DeviceBuffer::allocate(session, total_len.max(1))?;
        let mut slots = Vec::with_capacity(specs.len());
        for (spec, offset) in specs.iter().zip(offsets.into_iter()) {
            let subspan = buffer.subspan(offset, spec.byte_len)?;
            let view = subspan.create_view(
                &spec.shape,
                spec.element_type,
                ffi::iree_hal_encoding_types_t_IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR.0,
            )?;
            slots.push(ArenaSlot {
                offset,
                byte_len: spec.byte_len,
                _subspan: subspan,
                view,
            });
        }

        Ok(Self {
            buffer,
            slots,
            upload_staging: vec![0u8; total_len],
            download_staging: vec![0u8; total_len],
        })
    }

    pub fn view(&self, index: usize) -> &BufferView {
        &self.slots[index].view
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn upload_all(&mut self, session: &Session, host_slices: &[&[u8]]) -> Result<()> {
        if host_slices.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "host_slices length {} does not match arena slots {}",
                host_slices.len(),
                self.slots.len()
            )));
        }
        for (i, src) in host_slices.iter().enumerate() {
            self.write_slot_to_staging(i, src)?;
        }
        self.buffer.upload(session, &self.upload_staging, 0)
    }

    pub fn upload_selected(
        &mut self,
        session: &Session,
        selections: &[(usize, &[u8])],
    ) -> Result<()> {
        for (index, src) in selections {
            self.write_slot_to_staging(*index, src)?;
        }
        self.buffer.upload(session, &self.upload_staging, 0)
    }

    pub fn write_slot(&mut self, index: usize, src: &[u8]) -> Result<()> {
        self.write_slot_to_staging(index, src)
    }

    pub fn upload_staging(&self, session: &Session) -> Result<()> {
        self.buffer.upload(session, &self.upload_staging, 0)
    }

    pub fn copy_slot_from_view(
        &self,
        session: &Session,
        slot_index: usize,
        source_view: &BufferView,
    ) -> Result<()> {
        let slot = &self.slots[slot_index];
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let status = unsafe {
            ffi::iree_hal_device_transfer_d2d(
                session.device(),
                source_view.buffer_ptr(),
                0,
                self.buffer.ptr,
                slot.offset as ffi::iree_device_size_t,
                slot.byte_len as ffi::iree_device_size_t,
                ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT.0,
                timeout,
            )
        };
        error::check(status)
    }

    pub fn download_all_into(
        &mut self,
        session: &Session,
        host_slices: &mut [&mut [u8]],
    ) -> Result<()> {
        if host_slices.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "host_slices length {} does not match arena slots {}",
                host_slices.len(),
                self.slots.len()
            )));
        }
        self.buffer
            .download(session, &mut self.download_staging, 0)?;
        for (i, dst) in host_slices.iter_mut().enumerate() {
            let slot = &self.slots[i];
            if dst.len() != slot.byte_len {
                return Err(error::Error::invalid_argument(format!(
                    "host slice {} length {} does not match arena slot length {}",
                    i,
                    dst.len(),
                    slot.byte_len
                )));
            }
            let begin = slot.offset;
            let end = begin + slot.byte_len;
            dst.copy_from_slice(&self.download_staging[begin..end]);
        }
        Ok(())
    }

    pub fn download_all(&mut self, session: &Session) -> Result<()> {
        self.buffer.download(session, &mut self.download_staging, 0)
    }

    pub fn copy_slot_to_host(&self, index: usize, target: &mut [u8]) -> Result<()> {
        let slot = &self.slots[index];
        if target.len() != slot.byte_len {
            return Err(error::Error::invalid_argument(format!(
                "target length {} does not match arena slot length {}",
                target.len(),
                slot.byte_len
            )));
        }
        let begin = slot.offset;
        let end = begin + slot.byte_len;
        target.copy_from_slice(&self.download_staging[begin..end]);
        Ok(())
    }

    fn write_slot_to_staging(&mut self, index: usize, src: &[u8]) -> Result<()> {
        let slot = &self.slots[index];
        if src.len() != slot.byte_len {
            return Err(error::Error::invalid_argument(format!(
                "host slice {} length {} does not match arena slot length {}",
                index,
                src.len(),
                slot.byte_len
            )));
        }
        let begin = slot.offset;
        let end = begin + slot.byte_len;
        self.upload_staging[begin..end].copy_from_slice(src);
        Ok(())
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    let mask = alignment - 1;
    (value + mask) & !mask
}
