#![doc = include_str!("README.md")]

mod fft;
mod fftfreq;
mod time_slice;
mod norm;
pub use fft::*;
pub use fftfreq::*;
pub use time_slice::*;
pub use norm::*;

use crate::{Context, Error, Expr};
use std::collections::HashMap;
use std::sync::Arc;

pub trait EqlFormula: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    fn parse(&self, formula: Arc<dyn EqlFormula>, recv: Expr, args: &[Expr])
    -> Result<Expr, Error>;

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        expr.to_column_name()
            .map(|name| format!("{}({})", self.name(), name))
    }

    fn to_qualified_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("{}({})", self.name(), expr.to_qualified_field()?))
    }

    fn to_field(&self, expr: &Expr) -> Result<String, Error> {
        Ok(format!("{}({})", self.name(), expr.to_field()?))
    }

    fn suggestions(&self, _expr: &Expr, _context: &Context) -> Vec<String> {
        Vec::new()
    }

    fn to_select_part(&self, expr: &Expr) -> Result<String, Error> {
        if let Some(column_name) = self.to_column_name(expr) {
            Ok(format!(
                "{} as '{}'",
                self.to_qualified_field(expr)?,
                column_name
            ))
        } else {
            self.to_qualified_field(expr)
        }
    }

    fn to_sql(&self, expr: &Expr, context: &Context) -> Result<String, Error> {
        Ok(format!(
            "select {} from {}",
            // self.name(),
            self.to_select_part(expr)?,
            expr.to_table()?
        ))
    }
}

/// A formula registry that allows dynamic registration and lookup of formulas
#[derive(Debug, Clone)]
pub struct FormulaRegistry {
    formulas: HashMap<String, Arc<dyn EqlFormula>>,
}

impl FormulaRegistry {
    /// Create a new empty formula registry
    pub fn new() -> Self {
        Self {
            formulas: HashMap::new(),
        }
    }

    /// Register a formula in the registry
    pub fn register<F: EqlFormula + 'static>(&mut self, formula: F) {
        let name = formula.name().to_string();
        self.formulas.insert(name, Arc::new(formula));
    }

    /// Get a formula by name
    pub fn get(&self, name: &str) -> Option<&Arc<dyn EqlFormula>> {
        self.formulas.get(name)
    }

    /// Get all registered formula names
    pub fn formula_names(&self) -> impl Iterator<Item = &str> {
        self.formulas.keys().map(|s| s.as_str())
    }

    /// Check if a formula is registered
    pub fn contains(&self, name: &str) -> bool {
        self.formulas.contains_key(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn EqlFormula>> {
        self.formulas.values()
    }
}

impl Default for FormulaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a default formula registry with all built-in formulas
pub fn create_default_registry() -> FormulaRegistry {
    let mut registry = FormulaRegistry::new();

    // Register all built-in formulas
    registry.register(Fft);
    registry.register(FftFreq);
    registry.register(Norm);
    registry.register(TimeSlice::Last(None));
    registry.register(TimeSlice::First(None));

    registry
}
