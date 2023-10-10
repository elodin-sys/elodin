use bevy::prelude::{Children, Parent, Plugin, PostStartup};
use bevy_ecs::{
    entity::Entity,
    query::{With, Without},
    system::{Query, ResMut, Resource},
};
use bevy_utils::HashSet;

use crate::{tree::Joint, EntityQuery};

#[derive(Clone, Resource)]
pub struct TopologicalSort(pub Vec<Link>);

#[derive(Clone)]
pub struct Link {
    pub parent: Option<Entity>,
    pub child: Entity,
}

pub fn sort_system(
    mut sort: ResMut<TopologicalSort>,
    children_query: Query<&Children, With<Joint>>,
    roots: Query<(EntityQuery, Entity, Option<&Children>), Without<Parent>>,
) {
    fn recurse(
        parent: Entity,
        entity: Entity,
        children: &Children,
        children_query: &Query<&Children, With<Joint>>,
        sort: &mut TopologicalSort,
        visited: &mut HashSet<Entity>,
    ) {
        if visited.contains(&entity) {
            return;
        }
        for child in children {
            if let Ok(children) = children_query.get(*child) {
                recurse(entity, *child, children, children_query, sort, visited);
            }
        }
        visited.insert(parent);
        sort.0.push(Link {
            parent: Some(parent),
            child: entity,
        })
    }
    let mut visited = HashSet::default();
    for (_, parent, children) in &roots {
        if let Some(children) = children {
            for child in children {
                if let Ok(children) = children_query.get(*child) {
                    recurse(
                        parent,
                        *child,
                        children,
                        &children_query,
                        &mut sort,
                        &mut visited,
                    );
                }
            }
        }
        sort.0.push(Link {
            parent: None,
            child: parent,
        })
    }

    sort.0.reverse()
}

pub struct TopologicalSortPlugin;

impl Plugin for TopologicalSortPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(TopologicalSort(vec![]))
            .add_systems(PostStartup, sort_system);
    }
}
