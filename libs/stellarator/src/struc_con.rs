use std::{future::Future, marker::PhantomData, pin::Pin, task::Poll};

use futures_lite::future;

use crate::{
    Error,
    util::{CancelToken, OneshotRx, oneshot},
};

pub trait Joinable<T> {
    fn join(self) -> impl Future<Output = Result<T, Error>> + Send + Sync;
    fn cancel(self) -> impl Future<Output = Result<(), Error>> + Send + Sync;
}

pub struct Thread<T> {
    handle: std::thread::JoinHandle<()>,
    cancel_token: CancelToken,
    rx: OneshotRx<T>,
}

pub fn thread<T, F>(f: F) -> Thread<T>
where
    T: Send + 'static,
    F: FnOnce(CancelToken) -> T + Send + 'static,
{
    ThreadBuilder::default().thread(f)
}

impl<T: Send + Sync> Joinable<T> for Thread<T> {
    fn join(self) -> impl Future<Output = Result<T, Error>> {
        JoinFuture {
            f: self.rx.wait(),
            phantom_data: PhantomData,
            cancel_token: self.cancel_token,
            handle: Some(self.handle),
        }
    }

    async fn cancel(self) -> Result<(), Error> {
        self.cancel_token.cancel();
        Ok(())
    }
}

#[pin_project::pin_project(PinnedDrop)]
pub struct JoinFuture<T, F> {
    #[pin]
    f: F,
    phantom_data: PhantomData<T>,
    handle: Option<std::thread::JoinHandle<()>>,
    cancel_token: CancelToken,
}

#[pin_project::pinned_drop]
impl<T, F> PinnedDrop for JoinFuture<T, F> {
    fn drop(self: Pin<&mut Self>) {
        let this = self.project();
        this.cancel_token.cancel();
    }
}

impl<T, F> Future for JoinFuture<T, F>
where
    F: Future<Output = Option<T>>,
{
    type Output = Result<T, Error>;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.f.poll(cx) {
            Poll::Ready(res) => {
                let _ = this.handle.take().unwrap().join();
                Poll::Ready(res.ok_or(Error::JoinFailed))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(feature = "tokio")]
pub fn tokio<T, F, Fut>(f: F) -> Thread<T>
where
    T: Send + 'static,
    F: FnOnce(CancelToken) -> Fut + Send + 'static,
    Fut: Future<Output = T>,
{
    ThreadBuilder::default().tokio(f)
}

pub fn stellar<T, F, Fut>(f: F) -> Thread<Option<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
{
    ThreadBuilder::default().stellar(f)
}

#[derive(Default)]
pub struct ThreadBuilder {
    cancel_token: Option<CancelToken>,
}

impl ThreadBuilder {
    pub fn cancel_token(mut self, cancel_token: CancelToken) -> Self {
        self.cancel_token = Some(cancel_token);
        self
    }

    pub fn thread<T, F>(self, f: F) -> Thread<T>
    where
        T: Send + 'static,
        F: FnOnce(CancelToken) -> T + Send + 'static,
    {
        let (tx, rx) = oneshot::<T>();
        let cancel_token = self.cancel_token.unwrap_or_default();
        let thread_cancel_token = cancel_token.clone();
        let handle = std::thread::spawn(move || {
            tx.send(f(thread_cancel_token));
        });
        Thread {
            handle,
            rx,
            cancel_token,
        }
    }

    pub fn stellar<T, F, Fut>(self, f: F) -> Thread<Option<T>>
    where
        T: Send + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        self.thread(|cancel| {
            crate::run(move || {
                future::race(async { Some(f().await) }, async move {
                    cancel.wait().await;
                    None
                })
            })
        })
    }

    pub fn stellar_with_cancel<T, F, Fut>(self, f: F) -> Thread<T>
    where
        T: Send + 'static,
        F: FnOnce(CancelToken) -> Fut + Send + 'static,
        Fut: Future<Output = T> + Send + 'static,
    {
        self.thread(|cancel| crate::run(move || f(cancel)))
    }

    #[cfg(feature = "tokio")]
    pub fn tokio<T, F, Fut>(self, f: F) -> Thread<T>
    where
        T: Send + 'static,
        F: FnOnce(CancelToken) -> Fut + Send + 'static,
        Fut: Future<Output = T>,
    {
        self.thread(|cancel| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(f(cancel))
        })
    }
}
