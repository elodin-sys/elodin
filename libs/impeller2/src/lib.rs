//! Impeller is a protocol for sending and receiving n-dimensional arrays and messages in real-time environments.
//! Impeller was originally designed for aerospace, but can be used anywhere were you need to exchange data at a high data rate.
//!
//! Impeller is designed with a few goals in mind:
//! 1. High-performance / low-resource overhead
//! 2. Deterministic runtime - impeller is meant to be used in safety critical systems, and so it should be fully deterministic
//!
//!
//! ## Packet Format
//! Impeller sends data through a series of packets that contain a header and the actual data. Each packet consists of the following fields:
//! - packet_ty - The type of data enclosed in a packet, a msg, table, or time series
//! - id - An identifier for the contents of the packet. For a msg this is the msg id, and for a table it is the id of the vtable
//! - req_id - An identifier for the request this packet is associated with. This allows you to implement request reply semantics.
//!   req_id is usually used as an incrementing counter for each request from a client. The server then responds to the request with a packet
//!   with the same id
//!
//! Impeller has two primary data structures: msgs and tables
//!
//! ## Tables
//!
//! Tables are sets of tensors (n-dimensional arrays) organized using entities and components.
//! Entities are best thought of as objects that emit telemetry (i.e tensors).
//! This could be a sensor, actuator, or anything that logically has values associated with it.
//! Components is the telemetry itself. For instance if you had a IMU it could have angular_velocity and acceleration components on it.
//! Components are identified with a [`types::ComponentId`], and have a fixed [`schema::Schema`] associated with them.
//!
//! The tables themselves are laid out into a series of fields. Each field is a tensor associated with an entity and component id. Each tensor is expected
//! to be aligned. Conceptually this is similar to a repr(C) struct where the field names are entity, component id pairs.
//!
//! ## Msgs
//!
//! Sometimes you have data that does not cleanly fit into a fixed-size tensor (i.e a command that contains a string).
//! For that use case you can send arbitrary bytes over the protocol. Messages are prefixed with a 2 byte ID that identifies the message
//! By convention most messages are [https://docs.rs/postcard/latest/postcard/](postcard) formatted, but this is not a technical requirement

#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod buf;
pub mod com_de;
pub mod component;
pub mod error;
pub mod registry;
pub mod schema;
pub mod types;
pub mod util;
pub mod vtable;

#[doc(hidden)]
pub mod nox_impls;
