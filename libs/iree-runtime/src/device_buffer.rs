use std::mem::MaybeUninit;

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
            min_alignment: 64,
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

pub struct BufferMapping {
    inner: ffi::iree_hal_buffer_mapping_t,
}

impl BufferMapping {
    pub fn as_slice(&self) -> &[u8] {
        if self.inner.contents.data.is_null() || self.inner.contents.data_length == 0 {
            return &[];
        }
        unsafe {
            std::slice::from_raw_parts(self.inner.contents.data, self.inner.contents.data_length)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.inner.contents.data.is_null() || self.inner.contents.data_length == 0 {
            return &mut [];
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                self.inner.contents.data,
                self.inner.contents.data_length,
            )
        }
    }

    pub fn len(&self) -> usize {
        self.inner.contents.data_length
    }

    pub fn is_empty(&self) -> bool {
        self.inner.contents.data_length == 0
    }
}

impl Drop for BufferMapping {
    fn drop(&mut self) {
        unsafe { ffi::iree_hal_buffer_unmap_range(&mut self.inner) };
    }
}

impl DeviceBuffer {
    pub fn map_read(&self) -> Result<BufferMapping> {
        let byte_len = self.byte_len();
        let mut mapping = MaybeUninit::<ffi::iree_hal_buffer_mapping_t>::uninit();
        let status = unsafe {
            ffi::iree_hal_buffer_map_range(
                self.ptr,
                ffi::iree_hal_mapping_mode_bits_t_IREE_HAL_MAPPING_MODE_SCOPED.0 as _,
                ffi::iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_READ.0 as _,
                0,
                byte_len as ffi::iree_device_size_t,
                mapping.as_mut_ptr(),
            )
        };
        error::check(status)?;
        Ok(BufferMapping {
            inner: unsafe { mapping.assume_init() },
        })
    }

    pub fn map_write(&self) -> Result<BufferMapping> {
        let byte_len = self.byte_len();
        let access = ffi::iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_WRITE.0
            | ffi::iree_hal_memory_access_bits_t_IREE_HAL_MEMORY_ACCESS_READ.0;
        let mut mapping = MaybeUninit::<ffi::iree_hal_buffer_mapping_t>::uninit();
        let status = unsafe {
            ffi::iree_hal_buffer_map_range(
                self.ptr,
                ffi::iree_hal_mapping_mode_bits_t_IREE_HAL_MAPPING_MODE_SCOPED.0 as _,
                access as _,
                0,
                byte_len as ffi::iree_device_size_t,
                mapping.as_mut_ptr(),
            )
        };
        error::check(status)?;
        Ok(BufferMapping {
            inner: unsafe { mapping.assume_init() },
        })
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
            let aligned = align_up(total_len, 64);
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
        // When IREE aliases the output into the same device buffer as the input arena,
        // the data is already in-place and the d2d copy is a no-op — safe to ignore.
        match error::check(status) {
            Ok(()) => Ok(()),
            Err(err) if err.is_overlap_copy_error() => Ok(()),
            Err(err) => Err(err),
        }
    }

    pub fn copy_slots_from_views_batched(
        &self,
        session: &Session,
        source_views: &[BufferView],
    ) -> Result<()> {
        if source_views.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "source_views length {} does not match arena slots {}",
                source_views.len(),
                self.slots.len()
            )));
        }
        let mut command_buffer: *mut ffi::iree_hal_command_buffer_t = std::ptr::null_mut();
        let create_status = unsafe {
            ffi::iree_hal_command_buffer_create(
                session.device(),
                ffi::iree_hal_command_buffer_mode_bits_t_IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT.0,
                ffi::iree_hal_command_category_bits_t_IREE_HAL_COMMAND_CATEGORY_TRANSFER.0,
                !0u64,
                0,
                &mut command_buffer,
            )
        };
        error::check(create_status)?;

        struct CommandBufferGuard(*mut ffi::iree_hal_command_buffer_t);
        impl Drop for CommandBufferGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { ffi::iree_hal_command_buffer_release(self.0) };
                }
            }
        }
        let command_buffer_guard = CommandBufferGuard(command_buffer);

        let begin_status = unsafe { ffi::iree_hal_command_buffer_begin(command_buffer) };
        error::check(begin_status)?;

        for (slot, source_view) in self.slots.iter().zip(source_views.iter()) {
            let source_ref = ffi::iree_hal_buffer_ref_t {
                buffer: source_view.buffer_ptr(),
                offset: 0,
                length: slot.byte_len as ffi::iree_device_size_t,
                ..Default::default()
            };
            let target_ref = ffi::iree_hal_buffer_ref_t {
                buffer: self.buffer.ptr,
                offset: slot.offset as ffi::iree_device_size_t,
                length: slot.byte_len as ffi::iree_device_size_t,
                ..Default::default()
            };
            let copy_status = unsafe {
                ffi::iree_hal_command_buffer_copy_buffer(
                    command_buffer,
                    source_ref,
                    target_ref,
                    ffi::iree_hal_copy_flag_bits_t_IREE_HAL_COPY_FLAG_NONE
                        .0
                        .into(),
                )
            };
            error::check(copy_status)?;
        }

        let end_status = unsafe { ffi::iree_hal_command_buffer_end(command_buffer) };
        error::check(end_status)?;

        let mut semaphore: *mut ffi::iree_hal_semaphore_t = std::ptr::null_mut();
        let semaphore_create_status = unsafe {
            ffi::iree_hal_semaphore_create(
                session.device(),
                0u64,
                0,
                ffi::iree_hal_semaphore_flag_bits_t_IREE_HAL_SEMAPHORE_FLAG_NONE
                    .0
                    .into(),
                &mut semaphore,
            )
        };
        error::check(semaphore_create_status)?;

        struct SemaphoreGuard(*mut ffi::iree_hal_semaphore_t);
        impl Drop for SemaphoreGuard {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { ffi::iree_hal_semaphore_release(self.0) };
                }
            }
        }
        let semaphore_guard = SemaphoreGuard(semaphore);

        let mut signal_semaphores = [semaphore];
        let mut signal_values = [1u64];
        let wait_list = ffi::iree_hal_semaphore_list_t {
            count: 0,
            semaphores: std::ptr::null_mut(),
            payload_values: std::ptr::null_mut(),
        };
        let signal_list = ffi::iree_hal_semaphore_list_t {
            count: signal_semaphores.len(),
            semaphores: signal_semaphores.as_mut_ptr(),
            payload_values: signal_values.as_mut_ptr(),
        };
        let binding_table = ffi::iree_hal_buffer_binding_table_t {
            count: 0,
            bindings: std::ptr::null(),
        };
        let execute_status = unsafe {
            ffi::iree_hal_device_queue_execute(
                session.device(),
                !0u64,
                wait_list,
                signal_list,
                command_buffer,
                binding_table,
                ffi::iree_hal_execute_flag_bits_t_IREE_HAL_EXECUTE_FLAG_NONE
                    .0
                    .into(),
            )
        };
        error::check(execute_status)?;

        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let wait_status = unsafe {
            ffi::iree_hal_semaphore_wait(
                semaphore,
                1,
                timeout,
                ffi::iree_hal_wait_flag_bits_e_IREE_HAL_WAIT_FLAG_DEFAULT
                    .0
                    .into(),
            )
        };
        error::check(wait_status)?;

        drop(semaphore_guard);
        drop(command_buffer_guard);
        Ok(())
    }

    pub fn copy_slots_from_views_direct(
        &self,
        session: &Session,
        source_views: &[BufferView],
    ) -> Result<()> {
        if source_views.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "source_views length {} does not match arena slots {}",
                source_views.len(),
                self.slots.len()
            )));
        }
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        for (slot, source_view) in self.slots.iter().zip(source_views.iter()) {
            let status = unsafe {
                ffi::iree_hal_device_transfer_d2d(
                    session.device(),
                    source_view.buffer_ptr(),
                    0,
                    self.buffer.ptr,
                    slot.offset as ffi::iree_device_size_t,
                    slot.byte_len as ffi::iree_device_size_t,
                    ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT
                        .0,
                    timeout,
                )
            };
            match error::check(status) {
                Ok(()) => {}
                Err(err) if err.is_overlap_copy_error() => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
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

struct MappedSlot {
    offset: usize,
    byte_len: usize,
    _subspan: DeviceBuffer,
    view: BufferView,
}

/// Like `DeviceArena` but uses `iree_hal_buffer_map_range` instead of
/// staging buffers + HAL transfers. Only safe on CPU backends where
/// device memory is host-visible.
pub struct MappedArena {
    buffer: DeviceBuffer,
    slots: Vec<MappedSlot>,
}

impl MappedArena {
    pub fn new(session: &Session, specs: &[BufferSpec]) -> Result<Self> {
        let mut total_len = 0usize;
        let mut offsets = Vec::with_capacity(specs.len());
        for spec in specs {
            let aligned = align_up(total_len, 64);
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
            slots.push(MappedSlot {
                offset,
                byte_len: spec.byte_len,
                _subspan: subspan,
                view,
            });
        }

        Ok(Self { buffer, slots })
    }

    pub fn view(&self, index: usize) -> &BufferView {
        &self.slots[index].view
    }

    /// Write all slots into device memory via a single mapped write.
    pub fn upload_slots(&self, slot_data: &[(usize, &[u8])]) -> Result<()> {
        let mut mapping = self.buffer.map_write()?;
        let buf = mapping.as_mut_slice();
        for &(slot_idx, src) in slot_data {
            let slot = &self.slots[slot_idx];
            if src.len() != slot.byte_len {
                return Err(error::Error::invalid_argument(format!(
                    "upload_slots: slot {} source length {} does not match slot length {}",
                    slot_idx,
                    src.len(),
                    slot.byte_len
                )));
            }
            buf[slot.offset..slot.offset + slot.byte_len].copy_from_slice(src);
        }
        Ok(())
    }

    /// Read a single slot from device memory via mapped read.
    pub fn download_slot(&self, index: usize, target: &mut [u8]) -> Result<()> {
        let slot = &self.slots[index];
        if target.len() != slot.byte_len {
            return Err(error::Error::invalid_argument(format!(
                "target length {} does not match arena slot length {}",
                target.len(),
                slot.byte_len
            )));
        }
        let mapping = self.buffer.map_read()?;
        target.copy_from_slice(&mapping.as_slice()[slot.offset..slot.offset + slot.byte_len]);
        Ok(())
    }

    pub fn download_all_into(&self, host_slices: &mut [&mut [u8]]) -> Result<()> {
        if host_slices.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "host_slices length {} does not match arena slots {}",
                host_slices.len(),
                self.slots.len()
            )));
        }
        let mapping = self.buffer.map_read()?;
        let buf = mapping.as_slice();
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
            dst.copy_from_slice(&buf[slot.offset..slot.offset + slot.byte_len]);
        }
        Ok(())
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
        match error::check(status) {
            Ok(()) => Ok(()),
            Err(err) if err.is_overlap_copy_error() => Ok(()),
            Err(err) => Err(err),
        }
    }

    pub fn copy_slots_from_views_direct(
        &self,
        session: &Session,
        source_views: &[BufferView],
    ) -> Result<()> {
        if source_views.len() != self.slots.len() {
            return Err(error::Error::invalid_argument(format!(
                "source_views length {} does not match arena slots {}",
                source_views.len(),
                self.slots.len()
            )));
        }
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        for (slot, source_view) in self.slots.iter().zip(source_views.iter()) {
            let status = unsafe {
                ffi::iree_hal_device_transfer_d2d(
                    session.device(),
                    source_view.buffer_ptr(),
                    0,
                    self.buffer.ptr,
                    slot.offset as ffi::iree_device_size_t,
                    slot.byte_len as ffi::iree_device_size_t,
                    ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT
                        .0,
                    timeout,
                )
            };
            match error::check(status) {
                Ok(()) => {}
                Err(err) if err.is_overlap_copy_error() => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }
}
