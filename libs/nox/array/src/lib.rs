#![cfg_attr(not(feature = "std"), no_std)]

use zerocopy::{Immutable, TryFromBytes};

#[derive(Clone, Copy, Debug)]
pub struct ArrayView<'a, T> {
    pub buf: &'a [T],
    pub shape: &'a [usize],
}

impl<'a, T> ArrayView<'a, T> {
    pub fn from_bytes_shape_unchecked(buf: &'a [u8], shape: &'a [usize]) -> Option<Self>
    where
        [T]: TryFromBytes + Immutable,
    {
        let count = shape.iter().product();
        let buf = <[T]>::try_ref_from_bytes_with_elems(buf, count).ok()?;
        Some(ArrayView { buf, shape })
    }

    pub fn from_buf_shape_unchecked(buf: &'a [T], shape: &'a [usize]) -> Self {
        ArrayView { buf, shape }
    }

    pub fn as_bytes(&self) -> &[u8] {
        // Safe because we're only reading the bytes and T is guaranteed to be properly aligned
        unsafe {
            core::slice::from_raw_parts(
                self.buf.as_ptr() as *const u8,
                core::mem::size_of_val(self.buf),
            )
        }
    }

    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    pub fn buf(&self) -> &[T] {
        self.buf
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "std")]
use std::fmt;

#[cfg(feature = "std")]
impl<T: fmt::Display> fmt::Display for ArrayView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Constants from ndarray for consistency
        const MANY_ELEM_LIMIT: usize = 500;
        const AXIS_LIMIT: usize = 6;
        const ROW_LIMIT: usize = 11;
        const COL_LIMIT: usize = 11;

        let no_limit = f.alternate() || self.len() < MANY_ELEM_LIMIT;

        // Helper function to format array slice
        fn format_slice<T: fmt::Display>(
            slice: &[T],
            f: &mut fmt::Formatter<'_>,
            limit: usize,
        ) -> fmt::Result {
            if slice.is_empty() {
                return Ok(());
            }

            let show_all = limit == usize::MAX;
            let edge = limit / 2;

            if show_all || slice.len() <= limit {
                // Show all elements
                write!(f, "{}", slice[0])?;
                for x in slice.iter().skip(1) {
                    write!(f, ", {}", x)?;
                }
            } else {
                // Show edges with ellipsis
                write!(f, "{}", slice[0])?;
                for x in slice.iter().skip(1) {
                    write!(f, ", {}", x)?;
                }

                write!(f, ", ...")?;
                for x in slice.iter().skip(slice.len() - edge) {
                    write!(f, ", {}", x)?;
                }
            }
            Ok(())
        }

        if self.is_empty() {
            write!(
                f,
                "{}{}",
                "[".repeat(self.shape.len()),
                "]".repeat(self.shape.len())
            )?;
            return Ok(());
        }

        // Special case for 0-dimensional array (scalar)
        if self.shape.is_empty() {
            return write!(f, "{}", self.buf[0]);
        }

        // Helper function to recursively format n-dimensional arrays
        fn format_recursive<T: fmt::Display>(
            view: &ArrayView<T>,
            f: &mut fmt::Formatter<'_>,
            depth: usize,
            no_limit: bool,
        ) -> fmt::Result {
            let limit = if no_limit {
                usize::MAX
            } else if depth == 0 {
                ROW_LIMIT
            } else if depth == 1 {
                COL_LIMIT
            } else {
                AXIS_LIMIT
            };

            if view.shape.len() == 1 {
                write!(f, "[")?;
                format_slice(view.buf, f, limit)?;
                write!(f, "]")?;
                return Ok(());
            }

            let len = view.shape[0];
            let sub_len: usize = view.shape[1..].iter().product();

            write!(f, "[")?;

            if len <= limit || no_limit {
                // Show all subarrays
                if let Some(first) = view.buf.chunks(sub_len).next() {
                    let subview = ArrayView {
                        buf: first,
                        shape: &view.shape[1..],
                    };
                    format_recursive(&subview, f, depth + 1, no_limit)?;
                }
                for chunk in view.buf.chunks(sub_len).skip(1) {
                    write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    let subview = ArrayView {
                        buf: chunk,
                        shape: &view.shape[1..],
                    };
                    format_recursive(&subview, f, depth + 1, no_limit)?;
                }
            } else {
                // Show edges with ellipsis
                let edge = limit / 2;

                // First edge
                for (i, chunk) in view.buf.chunks(sub_len).take(edge).enumerate() {
                    if i > 0 {
                        write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    }
                    let subview = ArrayView {
                        buf: chunk,
                        shape: &view.shape[1..],
                    };
                    format_recursive(&subview, f, depth + 1, no_limit)?;
                }

                // Ellipsis
                write!(f, ",\n{}", " ".repeat(depth + 1))?;
                write!(f, "...")?;

                // Last edge
                for chunk in view.buf.chunks(sub_len).skip(len - edge) {
                    write!(f, ",\n{}", " ".repeat(depth + 1))?;
                    let subview = ArrayView {
                        buf: chunk,
                        shape: &view.shape[1..],
                    };
                    format_recursive(&subview, f, depth + 1, no_limit)?;
                }
            }

            write!(f, "]")?;
            Ok(())
        }

        format_recursive(self, f, 0, no_limit)
    }
}
