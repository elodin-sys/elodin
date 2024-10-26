use crate::reactor::Completion;
use crate::Error;
use std::io;
use std::path::Path;

pub struct File {
    handle: crate::os::OwnedHandle,
}

impl File {
    pub async fn create<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut options = OpenOptions::default();
        options.write(true).create(true).truncate(true);
        Self::open_with(path, &options).await
    }

    async fn open_with<P: AsRef<Path>>(path: P, options: &OpenOptions) -> Result<Self, Error> {
        let handle = Completion::run(crate::reactor::ops::Open::new(
            path.as_ref().to_path_buf(),
            options,
        )?)
        .await?;
        Ok(File { handle })
    }

    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut options = OpenOptions::default();
        options.read(true);
        Self::open_with(path, &options).await
    }

    pub async fn read<B: crate::buf::IoBufMut>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(crate::reactor::ops::Read::new(
            self.handle.as_handle(),
            buf,
            None,
        ))
        .await
    }

    pub async fn write<B: crate::buf::IoBuf>(&self, buf: B) -> Result<(usize, B), Error> {
        Completion::run(crate::reactor::ops::Write::new(
            self.handle.as_handle(),
            buf,
            None,
        ))
        .await
    }

    pub async fn read_at<B: crate::buf::IoBufMut>(
        &self,
        buf: B,
        offset: u64,
    ) -> Result<(usize, B), Error> {
        Completion::run(crate::reactor::ops::Read::new(
            self.handle.as_handle(),
            buf,
            Some(offset),
        ))
        .await
    }

    pub async fn write_at<B: crate::buf::IoBuf>(
        &self,
        buf: B,
        offset: u64,
    ) -> Result<(usize, B), Error> {
        Completion::run(crate::reactor::ops::Write::new(
            self.handle.as_handle(),
            buf,
            Some(offset),
        ))
        .await
    }

    pub async fn try_clone(&self) -> Result<Self, Error> {
        Ok(Self {
            handle: self.handle.try_clone()?,
        })
    }
}

/// Options and flags which can be used to configure how a file is opened
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
    pub(crate) custom_flags: libc::c_int,
    pub(crate) mode: u32,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
            custom_flags: 0,
            mode: 0o666,
        }
    }
}
impl OpenOptions {
    /// Returns a new OpenOptions object.
    ///
    /// This function returns a new OpenOptions object that you can use to
    /// open or create a file with specific options if `open()` or `create()`
    /// are not appropriate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the option for read access
    pub fn read(&mut self, read: bool) -> &mut Self {
        self.read = read;
        self
    }

    /// Sets the option for write access
    pub fn write(&mut self, write: bool) -> &mut Self {
        self.write = write;
        self
    }

    /// Sets the option for append access
    pub fn append(&mut self, append: bool) -> &mut Self {
        self.append = append;
        self
    }

    /// Sets the option to allow truncating the file
    ///
    /// This will set the length of an existing file when opened
    pub fn truncate(&mut self, truncate: bool) -> &mut Self {
        self.truncate = truncate;
        self
    }

    /// Sets the option to create a new file, or open it if it already exists
    pub fn create(&mut self, create: bool) -> &mut Self {
        self.create = create;
        self
    }

    /// Sets the option to create a new file, failing if it already exists.
    pub fn create_new(&mut self, create_new: bool) -> &mut Self {
        self.create_new = create_new;
        self
    }

    /// Set some custom flags for the file
    pub fn custom_flags(&mut self, custom_flags: libc::c_int) -> &mut Self {
        self.custom_flags = custom_flags;
        self
    }

    /// Set the mode to open the file with
    pub fn mode(&mut self, mode: u32) -> &mut Self {
        self.mode = mode;
        self
    }

    pub(crate) fn access_mode(&self) -> io::Result<libc::c_int> {
        match (self.read, self.write, self.append) {
            (true, false, false) => Ok(libc::O_RDONLY),
            (false, true, false) => Ok(libc::O_WRONLY),
            (true, true, false) => Ok(libc::O_RDWR),
            (false, _, true) => Ok(libc::O_WRONLY | libc::O_APPEND),
            (true, _, true) => Ok(libc::O_RDWR | libc::O_APPEND),
            (false, false, false) => Err(io::Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    pub(crate) fn creation_mode(&self) -> io::Result<libc::c_int> {
        match (self.write, self.append) {
            (true, false) => {}
            (false, false) => {
                if self.truncate || self.create || self.create_new {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
            }
            (_, true) => {
                if self.truncate && !self.create_new {
                    return Err(io::Error::from_raw_os_error(libc::EINVAL));
                }
            }
        }

        Ok(match (self.create, self.truncate, self.create_new) {
            (false, false, false) => 0,
            (true, false, false) => libc::O_CREAT,
            (false, true, false) => libc::O_TRUNC,
            (true, true, false) => libc::O_CREAT | libc::O_TRUNC,
            (_, _, true) => libc::O_CREAT | libc::O_EXCL,
        })
    }

    pub async fn open(&self, path: impl AsRef<Path>) -> Result<File, Error> {
        File::open_with(path, self).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test;

    #[test]
    fn test_open_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test");

        test!(async move {
            let mut options = OpenOptions::default();
            options.create(true).read(true).write(true);
            let file = File::open_with(path, &options).await.unwrap();
            let buf: &'static [u8] = b"test";
            let (written, _) = file.write(buf).await.unwrap();
            assert_eq!(written, 4);
            let out_buf = vec![0u8; 4];
            let (n, out_buf) = file.read_at(out_buf, 0).await.unwrap();
            assert_eq!(n, 4);
            assert_eq!(&out_buf, buf);
        })
    }
}
