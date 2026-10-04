//! Microbench for `stellarator::sleep` accuracy vs an absolute deadline.
//!
//! Investigation I1 for the real-time pacing work: how much does a requested
//! short sleep overshoot on the actual stellarator runtime (maitake timer
//! wheel + poll/kqueue reactor), idle vs under CPU contention? This is the
//! authoritative measurement the pacing loop depends on (the libc `time.sleep`
//! proxy overstated it).
//!
//! Usage:
//!   cargo run -p stellarator --release --example sleep_bench -- [period_us] [iters] [load_threads]
//!
//! It paces with an absolute deadline (deadline += period; sleep until it),
//! exactly like the sim loop, and reports per-cycle oversleep (actual sleep -
//! requested sleep) plus how often a cycle ends up behind its deadline. A
//! `sleep-until(deadline - 300us) + busy-spin` variant is also measured to show
//! the precision a hybrid pacer reaches on the same runtime.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((q * sorted.len() as f64) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn report(label: &str, mut xs_ms: Vec<f64>) {
    xs_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = xs_ms.iter().sum::<f64>() / xs_ms.len().max(1) as f64;
    println!(
        "  {label:22} mean={mean:7.3}  p50={:7.3}  p95={:7.3}  p99={:7.3}  max={:8.3}  (ms)",
        percentile(&xs_ms, 0.50),
        percentile(&xs_ms, 0.95),
        percentile(&xs_ms, 0.99),
        percentile(&xs_ms, 1.0),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let period_us: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(667);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let load_threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let period = Duration::from_micros(period_us);

    let stop = Arc::new(AtomicBool::new(false));
    let mut handles = Vec::new();
    for _ in 0..load_threads {
        let stop = stop.clone();
        handles.push(thread::spawn(move || {
            let mut x: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                std::hint::black_box(x);
            }
        }));
    }

    println!("period={period_us}us  iters={iters}  load_threads={load_threads}\n");

    stellarator::run(|| async move {
        // Pure absolute-deadline pacing (mirrors the sim loop).
        let mut oversleep_ms = Vec::with_capacity(iters);
        let mut behind = 0usize;
        let mut deadline = Instant::now() + period;
        for _ in 0..iters {
            let now = Instant::now();
            if now >= deadline {
                behind += 1;
            } else {
                let requested = deadline - now;
                let start = Instant::now();
                stellarator::sleep(requested).await;
                let actual = start.elapsed();
                oversleep_ms.push((actual.saturating_sub(requested)).as_secs_f64() * 1e3);
            }
            deadline += period;
        }
        report("oversleep (sleep)", oversleep_ms);
        println!(
            "  behind-deadline: {behind} / {iters} ({:.1}%)\n",
            100.0 * behind as f64 / iters as f64
        );

        // Hybrid: sleep until deadline-300us, then busy-spin to the deadline.
        let margin = Duration::from_micros(300);
        let mut over_ms = Vec::with_capacity(iters);
        let mut behind2 = 0usize;
        let mut deadline = Instant::now() + period;
        for _ in 0..iters {
            let now = Instant::now();
            if now >= deadline {
                behind2 += 1;
            } else {
                let requested = deadline - now;
                let start = Instant::now();
                if let Some(sleep_dur) = requested.checked_sub(margin)
                    && !sleep_dur.is_zero()
                {
                    stellarator::sleep(sleep_dur).await;
                }
                while Instant::now() < deadline {
                    std::hint::spin_loop();
                }
                let actual = start.elapsed();
                over_ms.push((actual.saturating_sub(requested)).as_secs_f64() * 1e3);
            }
            deadline += period;
        }
        report("oversleep (sleep+spin)", over_ms);
        println!(
            "  behind-deadline: {behind2} / {iters} ({:.1}%)",
            100.0 * behind2 as f64 / iters as f64
        );
    });

    stop.store(true, Ordering::Relaxed);
    for h in handles {
        let _ = h.join();
    }
}
