use std::ops::Range;

use bevy::math::{DVec3, Vec3};
use roaring::RoaringBitmap;

pub trait Unloaded: Clone {
    fn is_unloaded(&self) -> bool;
    fn unloaded() -> Self;
}

const UNLOADED: u64 = 0b111111111111000000000000000000000000000000000000000000011111111;
const UNLOADED_F32: u32 = 0b1111111110000000000000000011111;
impl Unloaded for f64 {
    fn is_unloaded(&self) -> bool {
        self.to_bits() == UNLOADED
    }

    #[inline(always)]
    fn unloaded() -> Self {
        f64::from_bits(UNLOADED)
    }
}

impl Unloaded for DVec3 {
    fn is_unloaded(&self) -> bool {
        self.x.to_bits() == UNLOADED && self.y.to_bits() == UNLOADED && self.z.to_bits() == UNLOADED
    }

    #[inline(always)]
    fn unloaded() -> Self {
        DVec3::new(
            f64::from_bits(UNLOADED),
            f64::from_bits(UNLOADED),
            f64::from_bits(UNLOADED),
        )
    }
}

impl Unloaded for Vec3 {
    fn is_unloaded(&self) -> bool {
        self.x.is_nan() && self.y.is_nan() && self.z.is_nan()
    }

    #[inline(always)]
    fn unloaded() -> Self {
        Vec3::new(f32::NAN, f32::NAN, f32::NAN)
    }
}

impl Unloaded for f32 {
    fn is_unloaded(&self) -> bool {
        self.to_bits() == UNLOADED_F32
    }

    #[inline(always)]
    fn unloaded() -> Self {
        f32::from_bits(UNLOADED_F32)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Chunk<T: Unloaded> {
    pub range: Range<usize>,
    pub data: Vec<T>,
    pub unfetched: RoaringBitmap,
}

impl<T: Unloaded> Chunk<T> {
    pub fn unhydrated(range: Range<usize>) -> Self {
        Chunk {
            data: range.clone().map(|_| T::unloaded()).collect(),
            range: range.clone(),
            unfetched: range.map(|x| x as u32).collect(),
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Chunks<T: Unloaded> {
    chunks: Vec<Option<Chunk<T>>>,
}

pub const CHUNK_SIZE: usize = 0x1000;

impl<T: Unloaded> Chunks<T> {
    pub fn range(&mut self, range: Range<usize>) -> impl Iterator<Item = &mut T> {
        self.chunks_range(range.clone()).flat_map(move |c| {
            let start = range.start.max(c.range.start) - c.range.start;
            let end = range.end.min(c.range.end) - c.range.start;
            &mut c.data[start..end]
        })
    }

    pub fn chunks_range(
        &mut self,
        requested_range: Range<usize>,
    ) -> impl Iterator<Item = &mut Chunk<T>> {
        let start = requested_range.start / CHUNK_SIZE;
        let end = requested_range.end / CHUNK_SIZE;
        let range = start..=end;
        if *range.start() >= self.chunks.len() {
            for _ in self.chunks.len()..=*range.start() {
                self.chunks.push(None);
            }
        }
        if *range.end() >= self.chunks.len() {
            for _ in self.chunks.len()..=*range.end() {
                self.chunks.push(None);
            }
        }
        self.chunks
            .get_mut(range.clone())
            .expect("vec did not contain range")
            .iter_mut()
            .enumerate()
            .map(move |(i, c)| {
                let start = (start + i) * CHUNK_SIZE;
                let end = start + CHUNK_SIZE;
                c.get_or_insert_with(|| Chunk::unhydrated(start..end))
            })
    }

    pub fn push(&mut self, tick: usize, value: T) {
        if let Some(Some(ref mut last)) = self.chunks.last_mut() {
            if last.range.end.saturating_add(1) == tick {
                let i = tick - last.range.start;
                if i > last.data.len() {
                    for j in last.data.len()..i {
                        last.data.insert(j, T::unloaded());
                        last.unfetched.insert(j as u32);
                    }
                }
                if i == last.data.len() {
                    last.data.push(value);
                } else {
                    last.data[i] = value;
                }

                last.unfetched.remove(i as u32);
                last.range.end = tick;
                return;
            }
        }
        for chunk in self.chunks_range(tick..tick + 1) {
            if chunk.range.contains(&tick) {
                let Some(i) = tick.checked_sub(chunk.range.start) else {
                    continue;
                };
                if i == chunk.data.len() {
                    chunk.data.push(value);
                } else {
                    chunk.data[i] = value;
                }
                chunk.unfetched.remove(i as u32);
                return;
            }
        }
    }

    pub fn get(&self, tick: usize) -> Option<&T> {
        let chunk = tick / CHUNK_SIZE;
        // let start = requested_range.start / CHUNK_SIZE;
        // let end = requested_range.end / CHUNK_SIZE;
        // let range = start..=end;
        let chunk = self.chunks.get(chunk)?.as_ref()?;
        let i = tick - chunk.range.start;
        chunk.data.get(i)
    }

    pub fn set_unfetched(&mut self, range: Range<usize>) {
        for chunk in self.chunks_range(range.clone()) {
            let start = range.start.max(chunk.range.start) as u32;
            let end = range.end.max(chunk.range.end) as u32;
            chunk.unfetched.insert_range(start..end);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShadowBuffer<T: Unloaded> {
    buf: Vec<T>,
    range: Range<usize>,
}

impl<T: Unloaded> Default for ShadowBuffer<T> {
    fn default() -> Self {
        Self {
            buf: Vec::new(),
            range: 0..0,
        }
    }
}

impl<T: Unloaded> ShadowBuffer<T> {
    pub fn new(buf: Vec<T>, range: Range<usize>) -> Self {
        Self { buf, range }
    }

    pub fn push(&mut self, tick: usize, value: T) -> bool {
        if tick >= self.range.end + 50 || tick < self.range.start {
            return false;
        }
        let tick = tick - self.range.start;
        if tick >= self.buf.len() {
            for _ in self.buf.len()..=tick {
                self.buf.push(T::unloaded());
            }
        }
        self.range.end = self.range.end.max(tick + 1);
        self.buf[tick] = value;
        true
    }

    pub fn buf(&self) -> &[T] {
        &self.buf
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_item() {
        let mut data = Chunks::<f64>::default();
        data.push(0, 1.0);
        data.push(2, 2.0);
        data.push(3, 3.0);
        data.push(5, 5.0);
        let vec = data.range(0..6).map(|x| *x).collect::<Vec<_>>();
        assert_eq!(vec[0], 1.0);
        assert!(vec[1].is_unloaded());
        assert_eq!(vec[2], 2.0);
        assert_eq!(vec[3], 3.0);
        assert!(vec[4].is_unloaded());
        assert_eq!(vec[5], 5.0);
        data.push(0x2000 + 2, 128.0);
        let vec = data.range(0x1FFF..0x2003).map(|x| *x).collect::<Vec<_>>();
        assert!(vec[0].is_unloaded());
        assert!(vec[1].is_unloaded());
        assert!(vec[2].is_unloaded());
        assert_eq!(vec[3], 128.0);
        data.push(0x8000 + 2, -5.0);
        let vec = data.range(0x2003..0x8005).map(|x| *x).collect::<Vec<_>>();
        for (i, x) in vec.iter().enumerate() {
            match i {
                0x5FFF => assert_eq!(*x, -5.0),
                _ => assert!(x.is_unloaded(), "index: {}", i),
            }
        }

        let mut data = Chunks::<DVec3>::default();
        data.push(0x8000, DVec3::new(3.0, 4.0, 5.0));
        for (i, x) in data.range(0x0..0x8001).enumerate() {
            match i {
                0x8000 => assert_eq!(*x, DVec3::new(3.0, 4.0, 5.0)),
                _ => assert!(x.is_unloaded(), "index: {}", i),
            }
        }
    }
}
