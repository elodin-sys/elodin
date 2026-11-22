//! Worker pool that runs groups of independent calls (see [`crate::parallel`]). The calling
//! thread and the workers claim tasks from every open region, and a task may open a region of
//! its own: whoever opens a region can finish it alone, so nesting cannot deadlock.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// Below this much measured work a group runs on the calling thread: waking workers costs more.
const SERIAL_BELOW_NS: u64 = 20_000;

/// How long an idle worker polls for new work before it sleeps. Short, so idle workers leave the
/// CPU to the rest of the simulation cycle, which calls [`wake`] before its next groups.
const SPIN: Duration = Duration::from_micros(100);

/// Workers run JIT code, so they get the simulation thread's stack.
const WORKER_STACK: usize = 256 * 1024 * 1024;

/// One call of a group: `func(ctx)`, laid out as the JIT stores it.
#[repr(C)]
pub struct Task {
    func: extern "C" fn(*mut u8),
    ctx: *mut u8,
}

impl Task {
    pub fn new(func: extern "C" fn(*mut u8), ctx: *mut u8) -> Self {
        Self { func, ctx }
    }
}

struct Region {
    tasks: *const Task,
    len: usize,
    next: AtomicUsize,
    done: AtomicUsize,
    work_ns: AtomicU64,
}

// Safety: `tasks` lives in the frame of the thread that opened the region, which waits in `run`
// until every claimed task is done; later claims get an index past `len` and touch no task.
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    fn work(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.len {
                return;
            }
            // Safety: `i < len`, and the table outlives every claimed task, as noted on `Region`.
            let task = unsafe { &*self.tasks.add(i) };
            let start = Instant::now();
            (task.func)(task.ctx);
            self.work_ns
                .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    fn has_unclaimed(&self) -> bool {
        self.next.load(Ordering::Relaxed) < self.len
    }
}

struct Pool {
    open: Mutex<Vec<Arc<Region>>>,
    /// Bumped under `open`'s lock whenever there is new work, so spinning workers notice it
    /// without taking the lock.
    generation: AtomicU64,
    parked: AtomicUsize,
    unpark: Condvar,
    workers: AtomicUsize,
}

static POOL: Pool = Pool {
    open: Mutex::new(Vec::new()),
    generation: AtomicU64::new(0),
    parked: AtomicUsize::new(0),
    unpark: Condvar::new(),
    workers: AtomicUsize::new(0),
};

impl Pool {
    fn publish(&self, region: Option<Arc<Region>>) {
        let mut open = self.open.lock().unwrap();
        open.extend(region);
        self.generation.fetch_add(1, Ordering::Release);
        if self.parked.load(Ordering::Relaxed) > 0 {
            self.unpark.notify_all();
        }
    }

    fn retire(&self, region: &Arc<Region>) {
        self.open
            .lock()
            .unwrap()
            .retain(|open| !Arc::ptr_eq(open, region));
    }

    fn claimable(&self) -> Option<Arc<Region>> {
        let open = self.open.lock().unwrap();
        open.iter().find(|region| region.has_unclaimed()).cloned()
    }

    fn worker(&self) {
        let mut seen = self.generation.load(Ordering::Acquire);
        loop {
            while let Some(region) = self.claimable() {
                region.work();
            }
            let idle_since = Instant::now();
            loop {
                let generation = self.generation.load(Ordering::Acquire);
                if generation != seen {
                    seen = generation;
                    break;
                }
                if idle_since.elapsed() < SPIN {
                    std::hint::spin_loop();
                    continue;
                }
                let open = self.open.lock().unwrap();
                if self.generation.load(Ordering::Acquire) == seen {
                    self.parked.fetch_add(1, Ordering::Relaxed);
                    drop(self.unpark.wait(open).unwrap());
                    self.parked.fetch_sub(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Grows the pool to `count` workers; the thread that runs a group works alongside them.
pub fn ensure_workers(count: usize) {
    static SPAWNING: Mutex<()> = Mutex::new(());
    let _spawning = SPAWNING.lock().unwrap();
    for index in POOL.workers.load(Ordering::Relaxed)..count {
        std::thread::Builder::new()
            .name(format!("elodin-worker-{index}"))
            .stack_size(WORKER_STACK)
            .spawn(|| POOL.worker())
            .expect("failed to spawn elodin worker thread");
        POOL.workers.store(index + 1, Ordering::Release);
    }
}

/// Worker threads already started. The calling thread is not included.
pub fn worker_count() -> usize {
    POOL.workers.load(Ordering::Acquire)
}

/// Wakes sleeping workers ahead of a group that is about to run.
pub fn wake() {
    if POOL.parked.load(Ordering::Relaxed) > 0 {
        POOL.publish(None);
    }
}

/// Runs `tasks[..len]` to completion. `estimate` keeps a moving average of the group's total
/// work in nanoseconds, which decides whether the next run uses the workers.
///
/// # Safety
///
/// `estimate` and `tasks[..len]` must be valid for the whole call, and each task's function must
/// be safe to call with its context from any thread.
pub unsafe extern "C" fn run(estimate: *const AtomicU64, tasks: *const Task, len: usize) {
    // Safety: the caller keeps `estimate` valid for the call.
    let estimate = unsafe { &*estimate };
    let previous = estimate.load(Ordering::Relaxed);
    let region = Region {
        tasks,
        len,
        next: AtomicUsize::new(0),
        done: AtomicUsize::new(0),
        work_ns: AtomicU64::new(0),
    };
    let serial =
        POOL.workers.load(Ordering::Relaxed) == 0 || (previous != 0 && previous < SERIAL_BELOW_NS);
    let work_ns = if serial {
        region.work();
        region.work_ns.into_inner()
    } else {
        let region = Arc::new(region);
        POOL.publish(Some(region.clone()));
        region.work();
        POOL.retire(&region);
        while region.done.load(Ordering::Acquire) < len {
            match POOL.claimable() {
                Some(other) => other.work(),
                None => std::hint::spin_loop(),
            }
        }
        region.work_ns.load(Ordering::Relaxed)
    };
    let updated = if previous == 0 {
        work_ns
    } else {
        (previous * 7 + work_ns) / 8
    };
    estimate.store(updated.max(1), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn add_one(ctx: *mut u8) {
        let cell = unsafe { &*(ctx as *const AtomicU64) };
        cell.fetch_add(1, Ordering::Relaxed);
    }

    extern "C" fn open_nested(ctx: *mut u8) {
        let tasks: Vec<Task> = (0..8).map(|_| Task { func: add_one, ctx }).collect();
        let estimate = AtomicU64::new(0);
        unsafe { run(&estimate, tasks.as_ptr(), tasks.len()) };
    }

    #[test]
    fn nested_regions_run_every_task() {
        ensure_workers(3);
        let counter = AtomicU64::new(0);
        let ctx = &counter as *const AtomicU64 as *mut u8;
        let tasks: Vec<Task> = (0..16)
            .map(|_| Task {
                func: open_nested,
                ctx,
            })
            .collect();
        let estimate = AtomicU64::new(0);
        for _ in 0..50 {
            unsafe { run(&estimate, tasks.as_ptr(), tasks.len()) };
        }
        assert_eq!(counter.load(Ordering::Relaxed), 50 * 16 * 8);
    }
}
