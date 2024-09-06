#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "posix")]
pub mod posix;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "kebab-case"))]
pub enum Task {
    Process(posix::Process),
    Watch(posix::Watcher),
}
