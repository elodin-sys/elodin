use crate::{ArrayElement, ElementType, Error, PrimitiveType, Result};
use cpp::{cpp, cpp_class};
use num_traits::FromPrimitive;
cpp_class!(pub unsafe struct RawShape as "Shape");

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ArrayShape {
    ty: ElementType,
    dims: Vec<i64>,
}

impl ArrayShape {
    /// Create a new array shape.
    pub fn new<E: ArrayElement>(dims: Vec<i64>) -> Self {
        Self { ty: E::TY, dims }
    }

    /// Create a new array shape.
    pub fn new_with_type(ty: ElementType, dims: Vec<i64>) -> Self {
        Self { ty, dims }
    }

    pub fn element_type(&self) -> ElementType {
        self.ty
    }

    pub fn ty(&self) -> ElementType {
        self.ty
    }

    /// The stored primitive type.
    pub fn primitive_type(&self) -> PrimitiveType {
        self.ty.primitive_type()
    }

    /// The number of elements stored in arrays that use this shape, this is the product of sizes
    /// across each dimension.
    pub fn element_count(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product::<usize>()
    }

    pub fn size(&self) -> usize {
        self.element_count() * self.ty().element_size_in_bytes()
    }

    pub fn dims(&self) -> &[i64] {
        &self.dims
    }

    pub fn first_dim(&self) -> Option<i64> {
        self.dims.first().copied()
    }

    pub fn last_dim(&self) -> Option<i64> {
        self.dims.last().copied()
    }
}

/// A shape specifies a primitive type as well as some array dimensions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Shape {
    Tuple(Vec<Shape>),
    Array(ArrayShape),
}

impl Shape {
    /// Create a new array shape.
    pub fn array<E: ArrayElement>(dims: Vec<i64>) -> Self {
        Self::Array(ArrayShape { ty: E::TY, dims })
    }

    /// Create a new array shape.
    pub fn array_with_type(ty: ElementType, dims: Vec<i64>) -> Self {
        Self::Array(ArrayShape { ty, dims })
    }

    /// Create a new tuple shape.
    pub fn tuple(shapes: Vec<Self>) -> Self {
        Self::Tuple(shapes)
    }

    /// The stored primitive type.
    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            Self::Tuple(_) => PrimitiveType::Tuple,
            Self::Array(a) => a.ty.primitive_type(),
        }
    }

    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            Self::Array { .. } => false,
        }
    }

    pub fn tuple_size(&self) -> Option<usize> {
        match self {
            Self::Tuple(shapes) => Some(shapes.len()),
            Self::Array { .. } => None,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Self::Tuple(shapes) => shapes.iter().map(|s| s.size()).sum(),
            Self::Array(a) => a.size(),
        }
    }

    pub fn raw_shape(&self) -> RawShape {
        match self {
            Self::Tuple(shapes) => {
                let shapes = shapes.iter().map(|s| s.raw_shape()).collect::<Vec<_>>();
                let shapes_ptr = shapes.as_ptr();
                let shapes_len = shapes.len();
                unsafe {
                    cpp!([shapes_ptr as "const Shape*", shapes_len as "size_t"] -> RawShape as "Shape" {
                        return ShapeUtil::MakeTupleShape(absl::Span(shapes_ptr, shapes_len));
                    })
                }
            }
            Self::Array(a) => {
                let dims_ptr = a.dims.as_ptr();
                let dims_len = a.dims.len();
                let prim_type = a.ty.primitive_type() as i32;
                if a.dims.iter().any(|x| *x < 0) {
                    unsafe {
                        cpp!([prim_type as "int32_t", dims_ptr as "const int64_t*", dims_len as "size_t"] -> RawShape as "Shape" {
                            std::vector<bool> dynamic;
                            std::vector<int64_t> bounds;
                            for (size_t i = 0; i < dims_len; ++i) {
                                if (dims_ptr[i] < 0) {
                                    bounds.push_back(-dims_ptr[i]);
                                    dynamic.push_back(true);
                                } else {
                                    bounds.push_back(dims_ptr[i]);
                                    dynamic.push_back(false);
                                }
                            }
                            return ShapeUtil::MakeShape(
                                (PrimitiveType)prim_type,
                                absl::Span<const int64_t>(bounds.data(), bounds.size()), dynamic);
                        })
                    }
                } else {
                    unsafe {
                        cpp!([prim_type as "int32_t", dims_ptr as "const int64_t*", dims_len as "size_t"] -> RawShape as "Shape" {
                            return ShapeUtil::MakeShape((PrimitiveType)prim_type, absl::Span<const int64_t>(dims_ptr, dims_len));
                        })
                    }
                }
            }
        }
    }
}

impl RawShape {
    pub fn primitive_type(&self) -> Result<PrimitiveType> {
        let ty = unsafe {
            cpp!([self as "const Shape*"] -> i32 as "int32_t" {
                return self->element_type();
            })
        };
        FromPrimitive::from_i32(ty).ok_or(Error::UnexpectedElementType(ty))
    }

    pub fn size(&self) -> usize {
        self.shape().unwrap().size()
    }

    pub fn shape(&self) -> Result<Shape> {
        let ty = self.primitive_type()?;
        match ty {
            PrimitiveType::Tuple => {
                let count = unsafe {
                    cpp!([self as "const Shape*"] -> usize as "size_t" {
                        return self->tuple_shapes_size();
                    })
                };
                let shapes = (0..count)
                    .map(|i| unsafe {
                        cpp!([self as "const Shape*", i as "size_t"] -> RawShape as "Shape" {
                            return self->tuple_shapes(i);
                        })
                    })
                    .map(|s| s.shape())
                    .collect::<Result<Vec<_>>>();
                shapes.map(Shape::Tuple)
            }

            ty => {
                let rank = unsafe {
                    cpp!([self as "const Shape*"] -> usize as "size_t" {
                        return self->dimensions_size();
                    })
                };

                let dims = (0..rank)
                    .map(|i| unsafe {
                        cpp!([self as "const Shape*", i as "size_t"] -> i64 as "int64_t" {
                            return self->dimensions(i);
                        })
                    })
                    .collect::<Vec<_>>();
                let ty = ty.element_type()?;
                Ok(Shape::Array(ArrayShape { ty, dims }))
            }
        }
    }
}
