use crate::admission::{self, AdmissionPermit};
use crate::cgroup::CgroupScope;
use crate::recipe::Recipe;
use clap::Parser;
use miette::miette;
use std::collections::HashMap;
use std::sync::Arc;
use stellarator::util::CancelToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RecipeExecution {
    Once,
    Watch,
}

impl RecipeExecution {
    fn from_watch(watch: bool) -> Self {
        if watch { Self::Watch } else { Self::Once }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    // The recipe to build
    recipe: Option<String>,
    #[arg(long)]
    release: bool,
    #[arg(long)]
    watch: bool,
}

impl Args {
    pub async fn run(&self, mut recipes: HashMap<String, Recipe>) -> miette::Result<()> {
        let recipe_name = self.recipe.clone().unwrap_or_else(|| "default".to_string());
        let recipe = recipes
            .remove(&recipe_name)
            .ok_or_else(|| miette!("{} recipe not found", recipe_name))?;
        run_recipe(recipe_name, recipe, self.watch, self.release).await
    }
}

/// Run a recipe, creating an internal CancelToken that is cancelled when the recipe completes.
pub async fn run_recipe(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
) -> miette::Result<()> {
    let cancel_token = CancelToken::new();
    let result =
        run_recipe_with_token(recipe_name, recipe, watch, release, cancel_token.clone()).await;
    cancel_token.cancel();
    result
}

/// Run a recipe with an externally provided CancelToken.
///
/// This allows the caller to control when the recipe is cancelled,
/// enabling graceful termination from outside the recipe's execution context.
///
/// Ctrl+C handling is still set up, so the recipe can be cancelled interactively.
pub async fn run_recipe_with_token(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
    cancel_token: CancelToken,
) -> miette::Result<()> {
    let admission_permit = admission::acquire_run_slot(admission::recipe_weight(&recipe)).await;
    run_recipe_with_token_admitted(
        recipe_name,
        recipe,
        watch,
        release,
        cancel_token,
        admission_permit,
    )
    .await
}

pub async fn run_recipe_with_token_admitted_in_cgroup(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
    cancel_token: CancelToken,
    admission_permit: Option<AdmissionPermit>,
    cgroup: Option<Arc<CgroupScope>>,
) -> miette::Result<()> {
    run_recipe_with_token_admitted_inner(
        recipe_name,
        recipe,
        watch,
        release,
        cancel_token,
        admission_permit,
        cgroup,
    )
    .await
}

pub async fn run_recipe_with_token_admitted(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
    cancel_token: CancelToken,
    admission_permit: Option<AdmissionPermit>,
) -> miette::Result<()> {
    run_recipe_with_token_admitted_inner(
        recipe_name,
        recipe,
        watch,
        release,
        cancel_token,
        admission_permit,
        None,
    )
    .await
}

async fn run_recipe_with_token_admitted_inner(
    recipe_name: String,
    recipe: Recipe,
    watch: bool,
    release: bool,
    cancel_token: CancelToken,
    _admission_permit: Option<AdmissionPermit>,
    cgroup: Option<Arc<CgroupScope>>,
) -> miette::Result<()> {
    // Set up Ctrl+C handling so the recipe can be cancelled interactively
    let ctrl_c_cancel_token = cancel_token.clone();
    tokio::spawn(async move {
        let _drop = ctrl_c_cancel_token.drop_guard();
        tokio::signal::ctrl_c().await
    });

    execute_recipe_with_token_in_cgroup(
        recipe_name,
        recipe,
        RecipeExecution::from_watch(watch),
        release,
        cancel_token,
        cgroup,
    )
    .await
}

/// Execute a recipe without adding admission control or signal handling.
///
/// This is the shared run-vs-watch dispatch used by normal headless/editor
/// execution and by the admitted wrapper used for Monte Carlo runs.
pub async fn execute_recipe_with_token_in_cgroup(
    recipe_name: String,
    recipe: Recipe,
    execution: RecipeExecution,
    release: bool,
    cancel_token: CancelToken,
    cgroup: Option<Arc<CgroupScope>>,
) -> miette::Result<()> {
    let result = match execution {
        RecipeExecution::Watch => {
            recipe
                .watch(recipe_name, release, cancel_token, cgroup)
                .await
        }
        RecipeExecution::Once => recipe.run(recipe_name, release, cancel_token, cgroup).await,
    };
    Ok(result?)
}

#[cfg(all(test, unix))]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use stellarator::util::CancelToken;

    use super::{RecipeExecution, execute_recipe_with_token_in_cgroup};
    use crate::{ProcessArgs, ProcessRecipe, Recipe, RestartPolicy};

    #[test]
    fn watch_flag_maps_to_shared_execution_mode() {
        assert_eq!(RecipeExecution::from_watch(false), RecipeExecution::Once);
        assert_eq!(RecipeExecution::from_watch(true), RecipeExecution::Watch);
    }

    #[tokio::test]
    async fn one_shot_execution_propagates_process_failure() {
        let recipe = Recipe::Process(ProcessRecipe {
            cmd: "sh".to_string(),
            process_args: ProcessArgs {
                args: vec!["-c".to_string(), "exit 7".to_string()],
                cwd: None,
                env: HashMap::new(),
                restart_policy: RestartPolicy::Never,
                fail_on_error: true,
                log_path: None,
                silence: false,
                depends_on: Vec::new(),
                ready: None,
                ready_timeout: None,
                own_process_group: false,
            },
            no_watch: true,
        });

        let result = tokio::time::timeout(
            Duration::from_secs(2),
            execute_recipe_with_token_in_cgroup(
                "test".to_string(),
                recipe,
                RecipeExecution::Once,
                false,
                CancelToken::new(),
                None,
            ),
        )
        .await
        .expect("one-shot execution waited instead of returning the process error");

        assert!(result.is_err());
    }
}
