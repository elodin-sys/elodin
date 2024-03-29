use std::marker::PhantomData;

use crate::{BufferForm, Client, Exec};

pub struct Comp<T, R> {
    pub comp: xla::XlaComputation,
    pub(crate) phantom: PhantomData<(T, R)>,
}

impl<T: BufferForm, R> Comp<T, R> {
    pub fn to_hlo_text(&self) -> Result<String, xla::Error> {
        self.comp.to_hlo_text()
    }

    pub fn compile(&self, client: &Client) -> Result<Exec<T::BufferTy, R>, xla::Error> {
        let exec = client.compile(&self.comp)?;
        Ok(Exec {
            exec,
            phantom: PhantomData,
        })
    }
}
