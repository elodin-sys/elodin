use futures_util::task::noop_waker;
use maitake::time::{Clock, Timer};
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

static NOW: AtomicU64 = AtomicU64::new(0);

fn now() -> u64 {
    NOW.load(Ordering::SeqCst)
}

#[test]
fn top_level_slot_zero_wrap_stays_in_the_next_rotation() {
    // Each top-level slot spans 2^30 ticks. Start halfway through slot 62,
    // then schedule two seconds ahead so the deadline lands in slot 0 of the
    // next rotation. This is the boundary that previously calculated a
    // deadline in the past and made Timer::turn spin until the timeout elapsed.
    const TOP_LEVEL_SLOT_TICKS: u64 = 1 << 30;
    const START: u64 = 62 * TOP_LEVEL_SLOT_TICKS + TOP_LEVEL_SLOT_TICKS / 2;
    const TIMEOUT_TICKS: u64 = 2_000_000_000;

    NOW.store(START, Ordering::SeqCst);
    let timer = Timer::new(Clock::new(Duration::from_nanos(1), now));
    timer.turn();

    let mut sleep = pin!(timer.sleep(Duration::from_secs(2)));
    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Pending);

    let turn = timer.turn();
    let until_next = turn
        .time_to_next_deadline()
        .expect("registered sleep should provide a deadline");
    assert!(until_next > Duration::ZERO);
    assert!(until_next <= Duration::from_secs(2));

    NOW.store(START + TIMEOUT_TICKS, Ordering::SeqCst);
    let turn = timer.turn();
    assert_eq!(turn.expired, 1);
    assert_eq!(sleep.as_mut().poll(&mut cx), Poll::Ready(()));
}
