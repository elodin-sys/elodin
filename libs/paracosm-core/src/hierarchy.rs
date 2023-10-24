use bevy::prelude::{Children, Parent, Plugin, PostStartup};
use bevy_ecs::{
    entity::Entity,
    query::{With, Without},
    system::{Query, ResMut, Resource},
};
use bevy_utils::HashSet;

use crate::{tree::Joint, EntityQuery, TreeIndex};

#[derive(Clone, Resource)]
pub struct TopologicalSort(pub Vec<Link>);

#[derive(Clone)]
pub struct Link {
    pub root: Entity,
    pub parent: Option<Entity>,
    pub child: Entity,
}

pub fn sort_system(
    mut sort: ResMut<TopologicalSort>,
    children_query: Query<&Children, With<Joint>>,
    mut index_query: Query<(&mut TreeIndex, &Joint)>,
    roots: Query<(EntityQuery, Entity, Option<&Children>), Without<Parent>>,
) {
    fn recurse(
        parent: Entity,
        entity: Entity,
        children: Option<&Children>,
        children_query: &Query<&Children, With<Joint>>,
        sort: &mut TopologicalSort,
        visited: &mut HashSet<Entity>,
        root: Entity,
    ) {
        if visited.contains(&entity) {
            return;
        }
        if let Some(children) = children {
            for child in children {
                recurse(
                    entity,
                    *child,
                    children_query.get(*child).ok(),
                    children_query,
                    sort,
                    visited,
                    root,
                );
            }
        }
        visited.insert(parent);
        sort.0.push(Link {
            parent: Some(parent),
            child: entity,
            root,
        })
    }
    let mut visited = HashSet::default();
    for (_, parent, children) in &roots {
        if let Some(children) = children {
            for child in children {
                recurse(
                    parent,
                    *child,
                    children_query.get(*child).ok(),
                    &children_query,
                    &mut sort,
                    &mut visited,
                    parent,
                );
            }
        }
        sort.0.push(Link {
            parent: None,
            child: parent,
            root: parent,
        })
    }

    sort.0.reverse();
    let mut i = 0;
    for link in sort.0.iter() {
        let Ok((mut index, joint)) = index_query.get_mut(link.child) else {
            continue;
        };
        index.0 = i;
        i += joint.dof();
    }
}

pub struct TopologicalSortPlugin;

impl Plugin for TopologicalSortPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(TopologicalSort(vec![]))
            .add_systems(PostStartup, sort_system);
    }
}
