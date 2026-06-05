use std::sync::{Arc, LazyLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Recipe;

pub type AdmissionPermit = OwnedSemaphorePermit;

static MAX_INFLIGHT: LazyLock<Option<usize>> = LazyLock::new(resolve_max_inflight);
static SEMAPHORE: LazyLock<Option<Arc<Semaphore>>> =
    LazyLock::new(|| MAX_INFLIGHT.map(|max| Arc::new(Semaphore::new(max))));

pub fn max_inflight() -> Option<usize> {
    *MAX_INFLIGHT
}

pub fn recipe_weight(recipe: &Recipe) -> usize {
    match recipe {
        Recipe::Cargo(_) | Recipe::Process(_) => 1,
        Recipe::Group(group) => group
            .recipes
            .values()
            .map(recipe_weight)
            .sum::<usize>()
            .max(1),
        #[cfg(not(target_os = "windows"))]
        Recipe::Sim(_) => 1,
    }
}

pub async fn acquire_run_slot(weight: usize) -> Option<AdmissionPermit> {
    let max = max_inflight()?;
    let semaphore = SEMAPHORE.as_ref()?.clone();
    Some(acquire_run_slot_with(semaphore, max, weight).await)
}

async fn acquire_run_slot_with(
    semaphore: Arc<Semaphore>,
    max: usize,
    weight: usize,
) -> AdmissionPermit {
    let permits = weight.max(1).min(max) as u32;
    semaphore
        .acquire_many_owned(permits)
        .await
        .expect("s10 admission semaphore is never closed")
}

fn resolve_max_inflight() -> Option<usize> {
    match std::env::var("S10_MAX_INFLIGHT") {
        Ok(raw) => {
            let raw = raw.trim();
            if raw.eq_ignore_ascii_case("off")
                || raw.eq_ignore_ascii_case("false")
                || raw.eq_ignore_ascii_case("none")
                || raw == "0"
            {
                return None;
            }
            raw.parse::<usize>().ok().filter(|max| *max > 0)
        }
        Err(_) => std::thread::available_parallelism().ok().map(usize::from),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::{GroupRecipe, ProcessArgs, ProcessRecipe, Recipe};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{Duration, timeout};

    fn process() -> Recipe {
        Recipe::Process(ProcessRecipe {
            cmd: "true".to_string(),
            process_args: ProcessArgs {
                args: vec![],
                cwd: None,
                env: HashMap::new(),
                restart_policy: crate::recipe::RestartPolicy::Never,
                fail_on_error: false,
                log_path: None,
            },
            no_watch: false,
        })
    }

    #[test]
    fn recipe_weight_counts_group_leaves() {
        let recipe = Recipe::Group(GroupRecipe {
            refs: vec![],
            recipes: HashMap::from([
                ("a".to_string(), process()),
                (
                    "nested".to_string(),
                    Recipe::Group(GroupRecipe {
                        refs: vec![],
                        recipes: HashMap::from([
                            ("b".to_string(), process()),
                            ("c".to_string(), process()),
                        ]),
                    }),
                ),
            ]),
        });

        assert_eq!(recipe_weight(&recipe), 3);
    }

    #[tokio::test]
    async fn acquire_run_slot_with_respects_cap() {
        let semaphore = Arc::new(Semaphore::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..6 {
            let semaphore = semaphore.clone();
            let active = active.clone();
            let peak = peak.clone();
            tasks.push(tokio::spawn(async move {
                let _permit = acquire_run_slot_with(semaphore, 2, 1).await;
                let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }

        for task in tasks {
            task.await.unwrap();
        }

        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[tokio::test]
    async fn oversized_single_run_does_not_deadlock() {
        let semaphore = Arc::new(Semaphore::new(2));
        let permit = timeout(
            Duration::from_millis(100),
            acquire_run_slot_with(semaphore, 2, 5),
        )
        .await
        .expect("oversized run should acquire capped permits");
        drop(permit);
    }
}
