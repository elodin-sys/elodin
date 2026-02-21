use crate::ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float16,
    Float32,
    Float64,
    BFloat16,
}

impl ElementType {
    pub(crate) fn to_hal(self) -> u32 {
        match self {
            Self::Bool => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BOOL_8.0,
            Self::Int8 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_8.0,
            Self::Int16 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_16.0,
            Self::Int32 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_32.0,
            Self::Int64 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_64.0,
            Self::Uint8 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_8.0,
            Self::Uint16 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_16.0,
            Self::Uint32 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_32.0,
            Self::Uint64 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_64.0,
            Self::Float16 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_16.0,
            Self::Float32 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_32.0,
            Self::Float64 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_64.0,
            Self::BFloat16 => ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BFLOAT_16.0,
        }
    }

    pub(crate) fn from_hal(val: u32) -> Option<Self> {
        let v = ffi::iree_hal_element_types_t(val);
        if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BOOL_8 {
            Some(Self::Bool)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_8 {
            Some(Self::Int8)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_16 {
            Some(Self::Int16)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_32 {
            Some(Self::Int32)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_SINT_64 {
            Some(Self::Int64)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_8 {
            Some(Self::Uint8)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_16 {
            Some(Self::Uint16)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_32 {
            Some(Self::Uint32)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_UINT_64 {
            Some(Self::Uint64)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_16 {
            Some(Self::Float16)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_32 {
            Some(Self::Float32)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_FLOAT_64 {
            Some(Self::Float64)
        } else if v == ffi::iree_hal_element_types_t_IREE_HAL_ELEMENT_TYPE_BFLOAT_16 {
            Some(Self::BFloat16)
        } else {
            None
        }
    }

    pub fn byte_count(self) -> usize {
        match self {
            Self::Bool | Self::Int8 | Self::Uint8 => 1,
            Self::Int16 | Self::Uint16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Int32 | Self::Uint32 | Self::Float32 => 4,
            Self::Int64 | Self::Uint64 | Self::Float64 => 8,
        }
    }
}
