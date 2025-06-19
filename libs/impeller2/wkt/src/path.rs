use std::fmt::Display;

use compact_str::CompactString;
use impeller2::types::ComponentId;
use smallvec::SmallVec;

#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ComponentPart {
    pub id: ComponentId,
    pub name: CompactString,
}

impl ComponentPart {
    pub fn new(name: &str) -> Self {
        Self {
            id: ComponentId::new(name),
            name: name.into(),
        }
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
#[cfg_attr(feature = "bevy", derive(bevy::prelude::Component))]
pub struct ComponentPath {
    pub id: ComponentId,
    pub path: SmallVec<[ComponentPart; 2]>,
}

impl std::hash::Hash for ComponentPath {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialOrd for ComponentPath {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ComponentPath {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl ComponentPath {
    pub fn from_name(name: &str) -> Self {
        let id = ComponentId::new(name);
        let path = name
            .match_indices('.')
            .map(|(i, _)| ComponentPart::new(&name[..i]))
            .chain([ComponentPart::new(name)])
            .collect();
        Self { id, path }
    }

    pub fn tail(&self) -> ComponentPart {
        let last = self.path.last().expect("empty path");
        let last = last.name.split(".").last().expect("missing last");
        ComponentPart::new(last)
    }

    pub fn is_top_level(&self) -> bool {
        self.path.len() == 1
    }
}

impl Display for ComponentPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, part) in self.path.iter().enumerate() {
            if i > 0 {
                write!(f, ".{}", part.name)?;
            } else {
                write!(f, "{}", part.name)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_path() {
        let path = ComponentPath::from_name("a.b.c");
        assert_eq!(
            &path.path[..],
            &[
                ComponentPart::new("a"),
                ComponentPart::new("a.b"),
                ComponentPart::new("a.b.c"),
            ]
        );
    }
}
