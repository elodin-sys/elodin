use std::sync::{Arc, OnceLock};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::Recipe;

pub type AdmissionPermit = OwnedSemaphorePermit;

/// Programmatic override of the admission budget (e.g. the monte-carlo
/// `workers` knob). Takes precedence over `S10_MAX_INFLIGHT` when set.
static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
/// Built lazily on first acquire so `configure` can still change the budget
/// any time before the first run is admitted.
static SEMAPHORE: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();

pub fn max_inflight() -> Option<usize> {
    match OVERRIDE.get() {
        Some(value) => *value,
        None => resolve_max_inflight(),
    }
}

/// Set the admission budget programmatically. Wins over `S10_MAX_INFLIGHT`.
/// Returns `false` when the budget is already locked in (a permit was
/// acquired, or a previous `configure` call happened first).
pub fn configure(max: Option<usize>) -> bool {
    if SEMAPHORE.get().is_some() {
        return false;
    }
    OVERRIDE.set(max).is_ok()
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
    let semaphore = SEMAPHORE
        .get_or_init(|| max_inflight().map(|max| Arc::new(Semaphore::new(max))))
        .clone()?;
    let max = max_inflight().unwrap_or(usize::MAX);
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
            parse_env_budget(raw)
        }
        Err(_) => std::thread::available_parallelism().ok().map(usize::from),
    }
}

/// The explicit numeric budget carried by `S10_MAX_INFLIGHT`, if any.
/// `"off"`-style and unparsable values yield `None`: they disable admission
/// limiting but express no budget, so callers deciding precedence (e.g. the
/// monte-carlo `workers` knob) must not treat them as an override.
pub fn env_budget() -> Option<usize> {
    parse_env_budget(&std::env::var("S10_MAX_INFLIGHT").ok()?)
}

fn parse_env_budget(raw: &str) -> Option<usize> {
    raw.trim().parse::<usize>().ok().filter(|max| *max > 0)
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
                silence: false,
                depends_on: Vec::new(),
                ready: None,
                ready_timeout: None,
                own_process_group: false,
            },
            no_watch: false,
        })
    }

    #[test]
    fn env_budget_only_accepts_positive_integers() {
        assert_eq!(parse_env_budget("96"), Some(96));
        assert_eq!(parse_env_budget(" 4 "), Some(4));
        assert_eq!(parse_env_budget("0"), None);
        assert_eq!(parse_env_budget("off"), None);
        assert_eq!(parse_env_budget("ninety-six"), None);
        assert_eq!(parse_env_budget(""), None);
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
