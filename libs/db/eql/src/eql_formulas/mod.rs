#![doc = include_str!("README.md")]

pub mod fft;
pub mod fftfreq;
pub mod first;
pub mod last;
pub mod norm;

use crate::{Expr, Error};

pub trait EqlFormula {
    fn name(&self) -> &'static str;

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error>;

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        expr.to_column_name().map(|name| format!("{}({name})", self.name()))
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("{}({})", self.name(), expr.to_qualified_field()?))
    }

    fn to_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("{}({})", self.name(), expr.to_field()?))
    }

    fn array_access_suggestion(&self) -> String {
        format!("{}()", self.name())
    }
}


#[derive(Debug, Clone)]
pub enum Formula {
    Fft(fft::Fft),
}

impl EqlFormula for Formula {
    fn name(&self) -> &'static str {
        match self {
            Formula::Fft(fft) => fft.name(),
        }
    }

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error>{
        match self {
            Formula::Fft(fft) => fft.parse(recv, args),
        }
    }

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        match self {
            Formula::Fft(fft) => fft.to_column_name(expr),
        }
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        match self {
            Formula::Fft(fft) => fft.to_qualified_field(expr),
        }
    }

    fn to_field(&self, expr: &Expr) -> Result<String, Error> {
        match self {
            Formula::Fft(fft) => fft.to_field(expr),
        }
    }

    fn array_access_suggestion(&self) -> String {
        match self {
            Formula::Fft(fft) => fft.array_access_suggestion(),
        }
    }
}
