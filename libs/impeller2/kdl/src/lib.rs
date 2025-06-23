use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

mod ser;
pub use ser::*;

mod de;
pub use de::*;

#[derive(Error, Debug, Diagnostic)]
pub enum KdlSchematicError {
    #[error("KDL parse error")]
    #[diagnostic(code(kdl_schematic::parse_error))]
    ParseError {
        #[source]
        source: kdl::KdlError,
        #[source_code]
        src: String,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Missing required property '{property}' on node '{node}'")]
    #[diagnostic(code(kdl_schematic::missing_property))]
    MissingProperty {
        property: String,
        node: String,
        #[source_code]
        src: String,
        #[label("this node is missing the property")]
        span: SourceSpan,
    },

    #[error("Invalid value for property '{property}' on node '{node}': expected {expected}")]
    #[diagnostic(code(kdl_schematic::invalid_value))]
    InvalidValue {
        property: String,
        node: String,
        expected: String,
        #[source_code]
        src: String,
        #[label("invalid value here")]
        span: SourceSpan,
    },

    #[error("Unknown node type '{node_type}'")]
    #[diagnostic(code(kdl_schematic::unknown_node))]
    UnknownNode {
        node_type: String,
        #[source_code]
        src: String,
        #[label("unknown node")]
        span: SourceSpan,
    },
}

pub trait ToKdl {
    fn to_kdl(&self) -> String;
}
pub trait FromKdl {
    fn from_kdl(src: &str) -> Result<Self, KdlSchematicError>
    where
        Self: Sized;
}

impl<T> ToKdl for impeller2_wkt::Schematic<T> {
    fn to_kdl(&self) -> String {
        serialize_schematic(self)
    }
}

impl FromKdl for impeller2_wkt::Schematic {
    fn from_kdl(src: &str) -> Result<Self, KdlSchematicError>
    where
        Self: Sized,
    {
        parse_schematic(src)
    }
}
