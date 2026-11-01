use impeller_kdl::{FromKdl, ToKdl};
use impeller_wkt::{Panel, Schematic, SchematicElem, Split};
use std::{future::Future, pin::Pin};

const MAX_WINDOWS: usize = 128;
const MAX_DEPTH: usize = 16;

// A WebGL device cannot present to another canvas's context. Use named tabs instead.
pub(super) async fn expand<F, Fut>(
    content: &str,
    key: &str,
    fetch: &mut F,
) -> Result<String, String>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<String, String>>,
{
    let root = Schematic::from_kdl(content).map_err(|error| error.to_string())?;
    if !root
        .elems
        .iter()
        .any(|elem| matches!(elem, SchematicElem::Window(_)))
    {
        return Ok(content.to_owned());
    }
    let key = resolve_key(key, "")?;
    let mut ancestors = vec![key.clone()];
    let mut remaining = MAX_WINDOWS;
    Ok(
        expand_schematic(root, &key, fetch, &mut ancestors, &mut remaining, 0)
            .await?
            .to_kdl(),
    )
}

fn expand_schematic<'a, F, Fut>(
    mut schematic: Schematic,
    key: &'a str,
    fetch: &'a mut F,
    ancestors: &'a mut Vec<String>,
    remaining: &'a mut usize,
    depth: usize,
) -> Pin<Box<dyn Future<Output = Result<Schematic, String>> + 'a>>
where
    F: FnMut(String) -> Fut + 'a,
    Fut: Future<Output = Result<String, String>> + 'a,
{
    Box::pin(async move {
        let mut elems = Vec::new();
        for elem in schematic.elems {
            let SchematicElem::Window(window) = elem else {
                elems.push(elem);
                continue;
            };
            if window.path.is_none() && window.content.is_none() {
                continue; // Desktop placement metadata has no browser content.
            }
            if depth >= MAX_DEPTH || *remaining == 0 {
                return Err("Too many nested schematic windows".to_owned());
            }
            *remaining -= 1;
            let child_key = window
                .path
                .as_deref()
                .map(|path| resolve_key(path, key))
                .transpose()?;
            let child = if let Some(child_key) = &child_key {
                if ancestors.contains(child_key) {
                    return Err(format!("Circular schematic window reference: {child_key}"));
                }
                let content = fetch(child_key.clone()).await?;
                let child = Schematic::from_kdl(&content)
                    .map_err(|error| format!("{child_key}: {error}"))?;
                ancestors.push(child_key.clone());
                child
            } else {
                window.content.unwrap_or_default()
            };
            let child = expand_schematic(
                child,
                child_key.as_deref().unwrap_or(key),
                fetch,
                ancestors,
                remaining,
                depth + 1,
            )
            .await?;
            if child_key.is_some() {
                ancestors.pop();
            }
            let title = window.title.or_else(|| {
                child_key
                    .as_deref()
                    .and_then(|key| key.rsplit('/').next())
                    .map(str::to_owned)
            });
            let panels = child
                .elems
                .into_iter()
                .filter_map(|elem| match elem {
                    SchematicElem::Panel(panel) => Some(panel),
                    _ => None,
                })
                .collect();
            elems.push(SchematicElem::Panel(Panel::HSplit(Split {
                panels: vec![Panel::Tabs(panels)],
                shares: Default::default(),
                active: false,
                name: Some(title.unwrap_or_else(|| "Window".to_owned())),
            })));
        }
        schematic.elems = elems;
        Ok(schematic)
    })
}

fn resolve_key(path: &str, parent: &str) -> Result<String, String> {
    let (path, base) = match path.strip_prefix("db:") {
        Some(key) => (key, ""),
        None => (path, parent.rsplit_once('/').map_or("", |(base, _)| base)),
    };
    if path.contains(':') || path.contains('\\') || path.starts_with('/') {
        return Err("Browser window paths must refer to recording assets".to_owned());
    }
    let mut parts: Vec<_> = base.split('/').filter(|part| !part.is_empty()).collect();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts
                    .pop()
                    .ok_or("Schematic window path leaves the recording")?;
            }
            part => parts.push(part),
        }
    }
    if parts.is_empty() {
        return Err("Empty schematic window path".to_owned());
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn expand_fixture(
        root: &str,
        assets: &[(&str, &str)],
    ) -> Result<(String, Vec<String>), String> {
        let assets: HashMap<_, _> = assets
            .iter()
            .map(|(key, content)| (key.to_string(), content.to_string()))
            .collect();
        let mut requests = Vec::new();
        let content = bevy::tasks::futures_lite::future::block_on(expand(
            root,
            "schematics/main.kdl",
            &mut |key| {
                requests.push(key.clone());
                let result = assets
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| format!("Missing asset: {key}"));
                async move { result }
            },
        ))?;
        Ok((content, requests))
    }

    #[test]
    fn nested_windows_become_named_tabs_and_keep_the_root_scene() {
        let root = "viewport name=\"Main\"\nwindow path=\"db:schematics/nested/detail.kdl\" title=\"Details\"";
        let assets = [
            (
                "schematics/nested/detail.kdl",
                "graph \"value\"\nwindow path=\"../inner #1.kdl\" title=\"Inner\"",
            ),
            ("schematics/inner #1.kdl", "viewport name=\"Nested view\""),
        ];
        let (expanded, requests) = expand_fixture(root, &assets).unwrap();
        assert_eq!(
            requests,
            ["schematics/nested/detail.kdl", "schematics/inner #1.kdl"]
        );
        let schematic = Schematic::from_kdl(&expanded).unwrap();
        assert!(
            schematic
                .elems
                .iter()
                .all(|elem| !matches!(elem, SchematicElem::Window(_)))
        );
        assert_eq!(schematic.viewports().len(), 2);
        assert!(expanded.contains("Details") && expanded.contains("Inner"));
        assert_eq!(expanded, expand_fixture(root, &assets).unwrap().0);
    }

    #[test]
    fn inline_and_repeated_windows_work_without_extra_surfaces() {
        let root = "window title=\"Inline\" { viewport; }\nwindow path=\"db:shared.kdl\"\nwindow path=\"db:shared.kdl\"";
        let (expanded, requests) = expand_fixture(root, &[("shared.kdl", "viewport")]).unwrap();
        assert_eq!(requests, ["shared.kdl", "shared.kdl"]);
        assert_eq!(Schematic::from_kdl(&expanded).unwrap().viewports().len(), 3);
        assert_eq!(expand_fixture("viewport", &[]).unwrap().0, "viewport");
    }

    #[test]
    fn circular_missing_and_invalid_windows_report_errors() {
        assert!(
            expand_fixture("window path=\"main.kdl\"", &[])
                .unwrap_err()
                .contains("Circular")
        );
        assert!(
            expand_fixture("window path=\"missing.kdl\"", &[])
                .unwrap_err()
                .contains("Missing asset")
        );
        assert!(
            expand_fixture(
                "window path=\"bad.kdl\"",
                &[("schematics/bad.kdl", "{invalid")]
            )
            .is_err()
        );
        for path in [
            "../../outside.kdl",
            "https://other.example/view.kdl",
            "/outside.kdl",
        ] {
            assert!(resolve_key(path, "schematics/main.kdl").is_err());
        }
        let root = std::iter::repeat_n("window path=\"db:shared.kdl\"", MAX_WINDOWS + 1)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            expand_fixture(&root, &[("shared.kdl", "viewport")])
                .unwrap_err()
                .contains("Too many")
        );
        let mut inline = "viewport".to_owned();
        for _ in 0..=MAX_DEPTH {
            inline = format!("window {{\n{inline}\n}}\n");
        }
        assert!(
            expand_fixture(&inline, &[])
                .unwrap_err()
                .contains("Too many")
        );
        assert!(
            expand_fixture(
                "window path=\"a.kdl\"",
                &[
                    ("schematics/a.kdl", "window path=\"b.kdl\""),
                    ("schematics/b.kdl", "window path=\"a.kdl\""),
                ]
            )
            .unwrap_err()
            .contains("Circular")
        );
    }
}
