use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("xla error {0}")]
    Xla(#[from] xla::Error),
}
