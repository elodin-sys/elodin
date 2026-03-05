use std::io::Write;
use std::sync::{Arc, Mutex};

/// Thread-safe buffer that captures all writes for reuse in tests.
#[allow(dead_code)]
pub struct SharedBuffer {
    inner: Arc<Mutex<Vec<u8>>>,
}

impl SharedBuffer {
    /// Creates a new shared buffer and returns it along with a handle to the
    /// stored bytes.
    #[allow(dead_code)]
    pub fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
        let inner = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                inner: inner.clone(),
            },
            inner,
        )
    }
}

impl Clone for SharedBuffer {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut guard = self.inner.lock().unwrap();
        guard.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Light-weight representation of an MP4 box used for parsing tests.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct Mp4Box {
    pub typ: [u8; 4],
    pub size: usize,
    pub offset: usize,
}

impl Mp4Box {
    /// Return the payload that immediately follows the box header.
    #[allow(dead_code)]
    pub fn payload<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        &data[self.offset + 8..self.offset + self.size]
    }
}

/// Parses top-level boxes from the provided MP4 data.
#[allow(dead_code)]
pub fn parse_boxes(data: &[u8]) -> Vec<Mp4Box> {
    let mut boxes = Vec::new();
    let mut cursor = 0;

    while cursor + 8 <= data.len() {
        let size = u32::from_be_bytes(data[cursor..cursor + 4].try_into().unwrap()) as usize;
        if size < 8 || cursor + size > data.len() {
            break;
        }
        let typ = data[cursor + 4..cursor + 8].try_into().unwrap();
        boxes.push(Mp4Box {
            typ,
            size,
            offset: cursor,
        });
        cursor += size;
    }

    boxes
}
