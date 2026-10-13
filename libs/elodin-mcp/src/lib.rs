//! Elodin's Code Mode MCP host: `search` over recording catalogs and `execute` of Python
//! analysis scripts, against local databases or an Elodin Cloud project.

pub mod cloud;
pub mod exec;
pub mod index;
pub mod search;
pub mod server;
pub mod source;
pub mod upload;
