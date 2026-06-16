use bevy::render::{
    render_resource::{encase::internal::WriteInto, *},
    renderer::{RenderDevice, RenderQueue},
};
use itertools::Itertools;
use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::Deref,
    path::{Path, PathBuf},
};

/// Resolve the single asset root used by the renderer.
///
/// This mirrors Elodin Editor's asset source convention: `ELODIN_ASSETS_DIR`
/// when set, otherwise `assets`, with relative paths resolved against the
/// current working directory.
pub fn assets_root() -> PathBuf {
    let mut root = std::env::var_os("ELODIN_ASSETS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets"));

    if root.is_relative() {
        if let Ok(cwd) = std::env::current_dir() {
            root = cwd.join(root);
        }
    }

    root
}

/// Resolve a path relative to [`assets_root`].
pub fn asset_path(relative: impl AsRef<Path>) -> PathBuf {
    assets_root().join(relative)
}

/// Resolve a path relative to [`assets_root`] and return a lossy string for
/// APIs that predate `PathBuf` plumbing.
pub fn asset_path_string(relative: impl AsRef<Path>) -> String {
    asset_path(relative).to_string_lossy().into_owned()
}

pub(crate) fn inverse_mix(a: f32, b: f32, value: f32) -> f32 {
    f32::clamp((value - a) / (b - a), 0.0, 1.0)
}

pub trait CollectArray: Iterator {
    fn collect_array<const T: usize>(self) -> [Self::Item; T]
    where
        Self: Sized,
        <Self as Iterator>::Item: Debug,
    {
        self.collect_vec().try_into().unwrap()
    }
}

impl<T> CollectArray for T where T: Iterator + ?Sized {}
enum Scratch {
    None,
    Uniform(encase::UniformBuffer<Vec<u8>>),
    Storage(encase::StorageBuffer<Vec<u8>>),
}

impl Scratch {
    fn new(usage: BufferUsages) -> Self {
        if usage.contains(BufferUsages::UNIFORM) {
            Self::Uniform(encase::UniformBuffer::new(Vec::new()))
        } else if usage.contains(BufferUsages::STORAGE) {
            Self::Storage(encase::StorageBuffer::new(Vec::new()))
        } else {
            Self::None
        }
    }

    fn write<T: ShaderType + WriteInto>(&mut self, value: &T) {
        match self {
            Scratch::None => panic!("Can't write to an buffer without a scratch buffer."),
            Scratch::Uniform(scratch) => scratch.write(value).unwrap(),
            Scratch::Storage(scratch) => scratch.write(value).unwrap(),
        }
    }

    fn contents(&self) -> &[u8] {
        match self {
            Scratch::None => panic!("Can't get the contents of a buffer without a scratch buffer."),
            Scratch::Uniform(scratch) => scratch.as_ref(),
            Scratch::Storage(scratch) => scratch.as_ref(),
        }
    }
}

pub struct StaticBuffer<T> {
    buffer: Buffer,
    value: Option<T>,
    scratch: Scratch,
    _marker: PhantomData<T>,
}

impl<T> StaticBuffer<T> {
    pub fn empty_sized<'a>(
        label: impl Into<Option<&'a str>>,
        device: &RenderDevice,
        size: BufferAddress,
        usage: BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: label.into(),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            value: None,
            scratch: Scratch::new(usage),
            _marker: PhantomData,
        }
    }

    pub fn update_bytes(&self, queue: &RenderQueue, bytes: &[u8]) {
        queue.write_buffer(&self.buffer, 0, bytes);
    }
}

impl<T: ShaderType + Default> StaticBuffer<T> {
    pub fn empty<'a>(
        label: impl Into<Option<&'a str>>,
        device: &RenderDevice,
        usage: BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: label.into(),
            size: T::min_size().get(),
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            value: None,
            scratch: Scratch::new(usage),
            _marker: PhantomData,
        }
    }
}

impl<T: ShaderType + WriteInto> StaticBuffer<T> {
    pub fn create<'a>(
        label: impl Into<Option<&'a str>>,
        device: &RenderDevice,
        value: &T,
        usage: BufferUsages,
    ) -> Self {
        let mut scratch = Scratch::new(usage);
        scratch.write(&value);

        let buffer = device.create_buffer_with_data(&BufferInitDescriptor {
            label: label.into(),
            usage,
            contents: scratch.contents(),
        });

        Self {
            buffer,
            value: None,
            scratch,
            _marker: PhantomData,
        }
    }

    pub fn value(&self) -> &T {
        self.value.as_ref().unwrap()
    }

    pub fn set_value(&mut self, value: T) {
        self.value = Some(value);
    }

    pub fn update(&mut self, queue: &RenderQueue) {
        if let Some(value) = &self.value {
            self.scratch.write(value);

            queue.write_buffer(&self.buffer, 0, self.scratch.contents());
        }
    }
}

impl<T> Deref for StaticBuffer<T> {
    type Target = Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl<'a, T> IntoBinding<'a> for &'a StaticBuffer<T> {
    #[inline]
    fn into_binding(self) -> BindingResource<'a> {
        self.buffer.as_entire_binding()
    }
}
