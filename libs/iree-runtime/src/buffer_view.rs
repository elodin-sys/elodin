use crate::DeviceBuffer;
use crate::element_type::ElementType;
use crate::error::{self, Result};
use crate::ffi;
use crate::ffi::iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT;
use crate::session::Session;

pub struct BufferView {
    pub(crate) ptr: *mut ffi::iree_hal_buffer_view_t,
}

impl BufferView {
    pub fn from_bytes(
        session: &Session,
        data: &[u8],
        shape: &[i64],
        element_type: ElementType,
    ) -> Result<Self> {
        let byte_span = ffi::iree_const_byte_span_t {
            data: data.as_ptr(),
            data_length: data.len(),
        };

        let shape_dims: Vec<ffi::iree_hal_dim_t> =
            shape.iter().map(|&d| d as ffi::iree_hal_dim_t).collect();

        let params = ffi::iree_hal_buffer_params_t {
            usage: ffi::iree_hal_buffer_usage_bits_t_IREE_HAL_BUFFER_USAGE_DEFAULT.0,
            access: 0,
            type_: ffi::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL.0
                | ffi::iree_hal_memory_type_bits_t_IREE_HAL_MEMORY_TYPE_HOST_VISIBLE.0,
            queue_affinity: 0,
            min_alignment: 0,
        };

        let mut view: *mut ffi::iree_hal_buffer_view_t = std::ptr::null_mut();
        let status = unsafe {
            ffi::iree_hal_buffer_view_allocate_buffer_copy(
                session.device(),
                session.device_allocator(),
                shape_dims.len(),
                shape_dims.as_ptr(),
                element_type.to_hal(),
                ffi::iree_hal_encoding_types_t_IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR.0,
                params,
                byte_span,
                &mut view,
            )
        };
        error::check(status)?;

        Ok(Self { ptr: view })
    }

    pub fn to_bytes(&self, session: &Session) -> Result<Vec<u8>> {
        let byte_length = unsafe { ffi::iree_hal_buffer_view_byte_length(self.ptr) };
        let mut bytes = vec![0u8; byte_length as usize];
        self.download_into(session, &mut bytes)?;
        Ok(bytes)
    }

    pub fn download_into(&self, session: &Session, target: &mut [u8]) -> Result<()> {
        let byte_length = self.byte_length();
        if target.len() != byte_length {
            return Err(error::Error::invalid_argument(format!(
                "target length {} does not match view byte length {}",
                target.len(),
                byte_length
            )));
        }
        let timeout = ffi::iree_timeout_t {
            type_: ffi::iree_timeout_type_e_IREE_TIMEOUT_ABSOLUTE,
            nanos: i64::MAX,
        };
        let status = unsafe {
            ffi::iree_hal_device_transfer_d2h(
                session.device(),
                self.buffer_ptr(),
                0,
                target.as_mut_ptr() as *mut std::ffi::c_void,
                target.len() as ffi::iree_device_size_t,
                iree_hal_transfer_buffer_flag_bits_t_IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT.0,
                timeout,
            )
        };
        error::check(status)
    }

    pub fn byte_length(&self) -> usize {
        unsafe { ffi::iree_hal_buffer_view_byte_length(self.ptr) as usize }
    }

    pub fn buffer(&self) -> DeviceBuffer {
        let ptr = self.buffer_ptr();
        unsafe { ffi::iree_hal_buffer_retain(ptr) };
        DeviceBuffer { ptr }
    }

    pub(crate) fn buffer_ptr(&self) -> *mut ffi::iree_hal_buffer_t {
        unsafe { ffi::iree_hal_buffer_view_buffer(self.ptr) }
    }

    pub fn shape(&self) -> Vec<i64> {
        let rank = unsafe { ffi::iree_hal_buffer_view_shape_rank(self.ptr) };
        (0..rank)
            .map(|i| unsafe { ffi::iree_hal_buffer_view_shape_dim(self.ptr, i) } as i64)
            .collect()
    }

    pub fn element_type(&self) -> Option<ElementType> {
        let hal_type = unsafe { ffi::iree_hal_buffer_view_element_type(self.ptr) };
        ElementType::from_hal(hal_type)
    }

    pub fn element_count(&self) -> usize {
        unsafe { ffi::iree_hal_buffer_view_element_count(self.ptr) as usize }
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::iree_hal_buffer_view_release(self.ptr) };
        }
    }
}
