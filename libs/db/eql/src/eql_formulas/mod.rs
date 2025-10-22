#![doc = include_str!("README.md")]

pub mod fft;
pub mod fftfreq;
pub mod first;
pub mod last;
pub mod norm;

use crate::{Error, Expr};
use std::collections::HashMap;
use std::sync::Arc;

// These default impls came from fft.
pub trait EqlFormula: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    fn parse(&self, recv: Expr, args: &[Expr]) -> Result<Expr, Error>;

    fn to_column_name(&self, expr: &Expr) -> Option<String> {
        expr.to_column_name()
            .map(|name| format!("{}({name})", self.name()))
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
    registry.register(fft::Fft);
    registry.register(fftfreq::FftFreq);
    registry.register(norm::Norm);
    registry.register(first::First);
    registry.register(last::Last);

    registry
}
