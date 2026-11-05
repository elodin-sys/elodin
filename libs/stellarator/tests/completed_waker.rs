use std::{
    future::poll_fn,
    sync::{Arc, Mutex},
    task::Poll,
};

#[stellarator::test]
async fn waking_the_last_reference_to_a_completed_task_releases_it_once() {
    let saved = Arc::new(Mutex::new(None));
    let output = saved.clone();
    let task = stellarator::spawn(poll_fn(move |cx| {
        *output.lock().unwrap() = Some(cx.waker().clone());
        Poll::Ready(())
    }));
    task.await.unwrap();
    // Transports can retain a waker after cancellation or EOF. Waking it by
    // value must release its final reference without decrementing it twice.
    saved.lock().unwrap().take().unwrap().wake();
}
