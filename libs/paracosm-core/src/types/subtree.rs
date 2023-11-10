use crate::spatial::SpatialInertia;
use bevy::prelude::*;
use nalgebra::{DMatrix, Vector3};

#[derive(Debug, Clone, PartialEq, Component)]
pub struct SubtreeInertia(pub SpatialInertia);
#[derive(Debug, Clone, PartialEq, Component)]
pub struct TreeIndex(pub usize);
#[derive(Debug, Clone, PartialEq, Resource)]
pub struct TreeMassMatrix(pub DMatrix<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct SubtreeMass(pub f64);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct SubtreeCoMSum(pub Vector3<f64>);
#[derive(Debug, Clone, Copy, PartialEq, Component, DerefMut, Deref)]
pub struct SubtreeCoM(pub Vector3<f64>);
