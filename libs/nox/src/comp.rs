use std::marker::PhantomData;

use crate::{BufferForm, Client, Exec};

pub struct Comp<T, R> {
    pub(crate) comp: xla::XlaComputation,
    pub(crate) phantom: PhantomData<(T, R)>,
}

impl<T: BufferForm, R> Comp<T, R> {
    pub fn compile(&self, client: &Client) -> Result<Exec<T::BufferTy, R>, xla::Error> {
        let exec = self.comp.compile(&client.0)?;
        Ok(Exec {
            exec,
            phantom: PhantomData,
        })
    }
}
