#![doc = include_str!("../README.md")]
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    str::FromStr,
    sync::Arc,
};
use unicode_ident::*;

use impeller2::{
    schema::Schema,
    types::{ComponentId, Timestamp},
};
use impeller2_wkt::ComponentPath;
use peg::error::ParseError;

pub mod formulas;

use formulas::{Formula, FormulaRegistry, create_default_registry};

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode<'input> {
    Ident(Cow<'input, str>),
    Field(Box<AstNode<'input>>, Cow<'input, str>),
    ArrayIndex(Box<AstNode<'input>>, usize),
    MethodCall(Box<AstNode<'input>>, Cow<'input, str>, Vec<AstNode<'input>>),
    BinaryOp(Box<AstNode<'input>>, Box<AstNode<'input>>, BinaryOp),
    Tuple(Vec<AstNode<'input>>),
    StringLiteral(Cow<'input, str>),
    FloatLiteral(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FmtNode<'input> {
    String(Cow<'input, str>),
    AstNode(AstNode<'input>),
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl BinaryOp {
    pub fn to_str(&self) -> &'static str {
        match self {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
        }
    }
}

peg::parser! {
    grammar ast_parser() for str {
        rule _ = quiet!{[' ' | '\n' | '\t']*}

        rule ident_str() -> Cow<'input, str>
            = i:$([c if is_xid_start(c)] [c if is_xid_continue(c)]*) { Cow::Borrowed(i) }

        rule uint() -> usize
            = s:$(['0'..='9']+) { s.parse().unwrap() }

        rule float() -> f64
            = s:$("-"? ['0'..='9']+ ("." ['0'..='9']*)?) { s.parse().unwrap() }

        rule string_literal() -> Cow<'input, str> = "\"" s:$([^'"']*) "\"" { Cow::Borrowed(s) }
        rule comma() = ("," _?)
        rule binary_op() -> BinaryOp = "+" { BinaryOp::Add } / "-" { BinaryOp::Sub } / "*" { BinaryOp::Mul } / "/" { BinaryOp::Div }

        rule fmt_ast_node() -> FmtNode<'input> = "${" e:expr() "}" { FmtNode::AstNode(e) }
        rule fmt_string_node() -> FmtNode<'input> = s:$([^'$']+) { FmtNode::String(Cow::Borrowed(s)) }
        rule fmt_node() -> FmtNode<'input> = fmt_ast_node() / fmt_string_node()
        pub rule fmt_string() -> Vec<FmtNode<'input>> = s:fmt_node()+ { s }

        pub rule expr() -> AstNode<'input> = precedence! {
        a:(@) comma() b:@ { AstNode::Tuple(vec![a, b]) }
        --
        a:(@) _ op:binary_op() _ b:@ { AstNode::BinaryOp(Box::new(a), Box::new(b), op) }
        --
        e:(@) "." i:ident_str() "(" args:expr() ** comma() ")" { AstNode::MethodCall(Box::new(e), i,  args) }
        --
        e:(@) "." i:ident_str() { AstNode::Field(Box::new(e), i) }
        --
        e:(@) "[" i:uint() "]" { AstNode::ArrayIndex(Box::new(e), i) }
        --
        "(" _ e:expr() _ ")" { e }
        --
        f:float() { AstNode::FloatLiteral(f) }
        --
        s:string_literal() { AstNode::StringLiteral(s) }
        --
        s:ident_str() { AstNode::Ident(s) }
        }
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    // core
    ComponentPart(Arc<ComponentPart>),
    Time(Arc<Component>),
    ArrayAccess(Box<Expr>, usize),
    Tuple(Vec<Expr>),
    FloatLiteral(f64),
    StringLiteral(String),

    Formula(Arc<dyn Formula>, Box<Expr>),

    // time limits
    Last(Box<Expr>, hifitime::Duration),
    First(Box<Expr>, hifitime::Duration),

    // infix
    BinaryOp(Box<Expr>, Box<Expr>, BinaryOp),
}

impl Expr {
    fn to_field(&self) -> Result<String, Error> {
        match self {
            Expr::ComponentPart(component) => Ok(component.name.replace(".", "_")),
            Expr::Time(_) => Ok("time".to_string()),
            Expr::Formula(formula, expr) => formula.to_field(expr),
            Expr::BinaryOp(left, right, op) => Ok(format!(
                "({} {} {})",
                left.to_field()?,
                op.to_str(),
                right.to_field()?
            )),

            Expr::ArrayAccess(inner_expr, index) => match inner_expr.as_ref() {
                Expr::ComponentPart(part) if part.component.is_some() => {
                    Ok(format!("{}[{}]", part.name.replace(".", "_"), index + 1))
                }
                _ => Err(Error::InvalidFieldAccess(
                    "array access on non-component".to_string(),
                )),
            },
            Expr::FloatLiteral(f) => Ok(format!("{}", f)),

            expr => Err(Error::InvalidFieldAccess(format!(
                "unsupported expression type for field {expr:?}"
            ))),
        }
    }

    fn to_table(&self) -> Result<String, Error> {
        match self {
            Expr::ComponentPart(component) => Ok(component.name.replace(".", "_")),

            Expr::Time(component) => Ok(component.name.replace(".", "_")),
            Expr::Formula(_, expr) => expr.to_table(),
            Expr::BinaryOp(left, _, _) => left.to_table(),

            Expr::ArrayAccess(inner_expr, _) => match inner_expr.as_ref() {
                Expr::ComponentPart(_) => inner_expr.to_table(),
                _ => Err(Error::InvalidFieldAccess(
                    "array access on non-component".to_string(),
                )),
            },

            Expr::Tuple(elements) => {
                // For tuples, find the first element that has a table (skip literals)
                for element in elements {
                    if let Ok(table) = element.to_table() {
                        return Ok(table);
                    }
                }
                Err(Error::InvalidFieldAccess(
                    "tuple contains no component references".to_string(),
                ))
            }

            expr => Err(Error::InvalidFieldAccess(format!(
                "unsupported expression type for table {expr:?}"
            ))),
        }
    }

    /// Converts an Expr to a qualified SQL field name (table.field) for use in JOINs.
    fn to_qualified_field(&self) -> Result<String, Error> {
        match self {
            Expr::Formula(formula, expr) => formula.to_qualified_field(expr),
            Expr::BinaryOp(left, right, op) => Ok(format!(
                "({} {} {})",
                left.to_qualified_field()?,
                op.to_str(),
                right.to_qualified_field()?
            )),
            Expr::FloatLiteral(f) => Ok(format!("{}", f)),
            _ => {
                let table = self.to_table()?;
                let field = self.to_field()?;
                Ok(format!("{}.{}", table, field))
            }
        }
    }

    fn to_column_name(&self) -> Option<String> {
        match self {
            Expr::Formula(formula, expr) => formula.to_column_name(expr),
            Expr::ComponentPart(e) => Some(e.name.clone()),
            Expr::ArrayAccess(expr, index) => match expr.as_ref() {
                Expr::ComponentPart(c) => {
                    if let Some(c) = &c.component {
                        c.element_names
                            .get(*index)
                            .map(|name| format!("{}.{}", c.name, name))
                    } else {
                        None
                    }
                }
                expr => Some(format!("{}[{}]", expr.to_column_name()?, index,)),
            },
            _ => None,
        }
    }

    fn to_select_part(&self) -> Result<String, Error> {
        if let Some(column_name) = self.to_column_name() {
            Ok(format!(
                "{} as '{}'",
                self.to_qualified_field()?,
                column_name
            ))
        } else {
            self.to_qualified_field()
        }
    }

    pub fn to_sql_time_field(&self) -> Result<String, Error> {
        match self {
            Expr::Tuple(elements) => {
                let Some(first) = elements.first() else {
                    return Err(Error::InvalidFieldAccess("empty tuple".to_string()));
                };
                first.to_sql_time_field()
            }
            expr => expr.to_table().map(|table| format!("{}.time", table)),
        }
    }

    /// Converts an EQL Expr to an SQL query string.
    pub fn to_sql(&self, context: &Context) -> Result<String, Error> {
        match self {
            Expr::Tuple(elements) => {
                if elements.is_empty() {
                    return Err(Error::InvalidFieldAccess("empty tuple".to_string()));
                }

                let mut table_names = BTreeSet::new();
                for element in elements {
                    table_names.insert(element.to_table()?);
                }

                if table_names.len() == 1 {
                    let mut select_parts = Vec::new();
                    for element in elements {
                        select_parts.push(element.to_select_part()?);
                    }
                    Ok(format!(
                        "select {} from {}",
                        select_parts.join(", "),
                        table_names.first().unwrap()
                    ))
                } else {
                    let mut select_parts = Vec::new();
                    for element in elements {
                        select_parts.push(element.to_select_part()?);
                    }
                    let first_table = table_names.first().unwrap();
                    let mut from_clause = table_names.first().unwrap().clone();

                    for table in table_names.iter().skip(1) {
                        from_clause.push_str(&format!(
                            " JOIN {} ON {}.time = {}.time",
                            table, first_table, table
                        ));
                    }

                    Ok(format!(
                        "select {} from {}",
                        select_parts.join(", "),
                        from_clause
                    ))
                }
            }
            Expr::BinaryOp(left, right, op) => {
                let left_table_name = left.to_table()?;
                let right_table_name = right.to_table()?;
                let left_select_part = left.to_qualified_field()?;
                let right_select_part = right.to_qualified_field()?;
                if left_table_name == right_table_name {
                    Ok(format!(
                        "select {} {} {} from {}",
                        left_select_part,
                        op.to_str(),
                        right_select_part,
                        left_table_name
                    ))
                } else {
                    Ok(format!(
                        "select {} {} {} from {} join {} on {}.time = {}.time",
                        left_select_part,
                        op.to_str(),
                        right_select_part,
                        left_table_name,
                        right_table_name,
                        left_table_name,
                        right_table_name
                    ))
                }
            }

            Expr::StringLiteral(_) => Err(Error::InvalidFieldAccess(
                "cannot convert string literal to SQL".to_string(),
            )),
            Expr::Formula(formula, expr) => formula.to_sql(expr, context),
            expr => Ok(format!(
                "select {} from {}",
                expr.to_select_part()?,
                expr.to_table()?
            )),
        }
    }
}

pub enum FmtExpr {
    String(String),
    Expr(Expr),
}

#[derive(Clone, Debug)]
pub struct ComponentPart {
    pub name: String,
    pub id: ComponentId,
    pub component: Option<Arc<Component>>,
    pub children: BTreeMap<String, ComponentPart>,
}

#[derive(Clone, Debug)]
pub struct Component {
    pub name: String,
    pub id: ComponentId,
    pub schema: Schema,
    pub element_names: Vec<String>,
}

impl Component {
    pub fn new(name: String, id: ComponentId, schema: Schema) -> Self {
        let element_names = default_element_names(schema.dim());
        Self {
            name,
            id,
            schema,
            element_names,
        }
    }
}

fn default_element_names(shape: &[u64]) -> Vec<String> {
    fn append_elements(shape: &[u64], parent_elem: &str, elems: &mut Vec<String>) {
        if shape.is_empty() {
            elems.push(parent_elem.to_string());
            return;
        }
        for x in 0..shape[0] {
            let mut elem = parent_elem.to_string();
            const ELEMS: [char; 8] = ['x', 'y', 'z', 'w', 'u', 'v', 's', 't'];
            if let Some(x) = ELEMS.get(x as usize) {
                elem.push(*x);
            } else {
                elem.push_str(&x.to_string());
            }
            append_elements(&shape[1..], &elem, elems);
        }
    }
    let mut elems = Vec::new();
    append_elements(shape, "", &mut elems);
    elems
}

#[derive(Debug)]
pub struct Context {
    pub component_parts: BTreeMap<String, ComponentPart>,
    pub earliest_timestamp: Timestamp,
    pub last_timestamp: Timestamp,
    pub formula_registry: FormulaRegistry,
}

impl Default for Context {
    fn default() -> Self {
        Context {
            component_parts: BTreeMap::new(),
            earliest_timestamp: Timestamp(i64::MIN),
            last_timestamp: Timestamp(i64::MAX),
            formula_registry: create_default_registry(),
        }
    }
}

impl Context {
    pub fn from_leaves(
        components: impl IntoIterator<Item = Arc<Component>>,
        earliest_timestamp: Timestamp,
        last_timestamp: Timestamp,
    ) -> Self {
        let mut component_parts = BTreeMap::new();
        for component in components.into_iter() {
            let path = ComponentPath::from_name(&component.name);
            let nodes = component.name.split('.');
            let mut component_parts = &mut component_parts;
            for (part, node) in path
                .path
                .iter()
                .zip(nodes)
                .take(path.path.len().saturating_sub(1))
            {
                let part =
                    component_parts
                        .entry(node.to_string())
                        .or_insert_with(|| ComponentPart {
                            id: part.id,
                            name: part.name.to_string(),
                            children: BTreeMap::new(),
                            component: None,
                        });
                component_parts = &mut part.children;
            }
            let mut nodes = component.name.split('.');
            component_parts.insert(
                nodes.next_back().unwrap().to_string(),
                ComponentPart {
                    id: component.id,
                    name: component.name.clone(),
                    children: BTreeMap::new(),
                    component: Some(component),
                },
            );
        }
        Self {
            component_parts,
            earliest_timestamp,
            last_timestamp,
            formula_registry: create_default_registry(),
        }
    }

    pub fn new(
        component_parts: BTreeMap<String, ComponentPart>,
        earliest_timestamp: Timestamp,
        last_timestamp: Timestamp,
    ) -> Self {
        Self {
            component_parts,
            earliest_timestamp,
            last_timestamp,
            formula_registry: create_default_registry(),
        }
    }

    pub fn sql(&self, query: &str) -> Result<String, Error> {
        let ast = ast_parser::expr(query)?;
        let expr = self.parse(&ast)?;
        expr.to_sql(self)
    }

    pub fn parse_str(&self, query: &str) -> Result<Expr, Error> {
        let ast = ast_parser::expr(query)?;
        self.parse(&ast)
    }

    pub fn parse(&self, ast: &AstNode) -> Result<Expr, Error> {
        match ast {
            AstNode::Ident(cow) => self
                .component_parts
                .get(cow.as_ref())
                .ok_or(Error::ComponentNotFound(cow.to_string()))
                .cloned()
                .map(Arc::new)
                .map(Expr::ComponentPart),
            AstNode::Field(ast_node, cow) => {
                let expr = self.parse(ast_node)?;
                match &expr {
                    Expr::ComponentPart(part) => {
                        if let Some(c) = &part.component {
                            match cow.as_ref() {
                                "time" => return Ok(Expr::Time(c.clone())),
                                _ => {
                                    if let Some(offset) =
                                        c.element_names.iter().position(|name| name == cow.as_ref())
                                    {
                                        return Ok(Expr::ArrayAccess(Box::new(expr), offset));
                                    }
                                }
                            }
                        }
                        part.children
                            .get(cow.as_ref())
                            .ok_or(Error::ComponentNotFound(cow.to_string()))
                            .cloned()
                            .map(Arc::new)
                            .map(Expr::ComponentPart)
                    }
                    _ => Err(Error::InvalidFieldAccess(cow.to_string())),
                }
            }
            AstNode::MethodCall(recv, cow, ast_nodes) => {
                let recv = self.parse(recv)?;
                let args = ast_nodes
                    .iter()
                    .map(|ast_node| self.parse(ast_node))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(formula) = self.formula_registry.get(cow.as_ref()) {
                    formula.parse(recv, &args)
                } else {
                    Err(Error::InvalidMethodCall(cow.to_string()))
                }
            }
            AstNode::Tuple(ast_nodes) => ast_nodes
                .iter()
                .map(|ast_node| self.parse(ast_node))
                .collect::<Result<Vec<_>, _>>()
                .map(Expr::Tuple),
            AstNode::StringLiteral(s) => Ok(Expr::StringLiteral(s.to_string())),
            AstNode::BinaryOp(left, right, op) => {
                let left = self.parse(left)?;
                let right = self.parse(right)?;
                Ok(Expr::BinaryOp(Box::new(left), Box::new(right), *op))
            }
            AstNode::FloatLiteral(f) => Ok(Expr::FloatLiteral(*f)),
            AstNode::ArrayIndex(ast_node, index) => {
                let expr = self.parse(ast_node)?;
                match &expr {
                    Expr::ComponentPart(_) => Ok(Expr::ArrayAccess(Box::new(expr), *index)),
                    _ => Err(Error::InvalidFieldAccess(
                        "array access is only valid on a component".to_string(),
                    )),
                }
            }
        }
    }

    pub fn parse_fmt_string(&self, input: &str) -> Result<Vec<FmtExpr>, Error> {
        let fmt_nodes = ast_parser::fmt_string(input)?;
        fmt_nodes
            .into_iter()
            .map(|node| {
                Ok(match node {
                    FmtNode::String(cow) => FmtExpr::String(cow.to_string()),
                    FmtNode::AstNode(ast_node) => FmtExpr::Expr(self.parse(&ast_node)?),
                })
            })
            .collect()
    }

    /// Get suggestions for the given expression
    pub fn get_suggestions(&self, expr: &Expr) -> Vec<String> {
        let mut suggestions: Vec<String> = match expr {
            Expr::ComponentPart(part) => {
                let mut suggestions: Vec<String> = part.children.keys().cloned().collect();
                if let Some(component) = &part.component {
                    suggestions.extend(component.element_names.iter().map(|name| name.to_string()));
                }
                suggestions.push("time".to_string());
                suggestions
            }
            Expr::StringLiteral(_) => Vec::new(),
            _ => Vec::new(),
        };

        for formula in self.formula_registry.iter() {
            suggestions.extend(formula.suggestions(expr, self));
        }

        suggestions.sort();
        suggestions.dedup();
        suggestions
    }

    pub fn get_string_suggestions(&self, input: &str) -> Vec<(String, String)> {
        fn apply_suggestions(
            suggestions: &[String],
            input: &str,
            start: &str,
        ) -> Vec<(String, String)> {
            suggestions
                .iter()
                .map(|s| (s.clone(), format!("{start}{input}.{s}")))
                .collect()
        }

        // strip tuple
        let valid_ast = ast_parser::expr(input.trim().trim_end_matches('.')).is_ok();
        let last_comma_pos = input.rfind(',');
        let last_paren_pos = input.rfind('(');
        let last_end_paren_pos = input.rfind(')');

        let (start, input) = match (
            valid_ast,
            last_comma_pos,
            last_paren_pos,
            last_end_paren_pos,
        ) {
            (true, Some(_), Some(start), Some(end)) if start < end => ("", input),
            (false, Some(comma_pos), Some(paren_pos), None) if comma_pos > paren_pos => {
                let comma_pos = input.rfind(',').unwrap();
                input.split_at(comma_pos + 1)
            }
            (_, Some(comma_pos), _, _) => input.split_at(comma_pos + 1),
            (_, _, Some(paren_pos), _) => input.split_at(paren_pos + 1),
            _ => ("", input),
        };

        if !input.contains(".") {
            let mut component_names = self.component_parts.keys().cloned().collect::<Vec<_>>();
            component_names.sort();
            return component_names
                .into_iter()
                .filter(|n| n.contains(input.trim()))
                .map(|n| (n.clone(), format!("{start}{n}")))
                .collect();
        }

        if let Some(query) = input.strip_suffix('.') {
            let Ok(ast) = ast_parser::expr(query.trim()) else {
                return vec![];
            };
            if let Ok(expr) = self.parse(&ast) {
                return apply_suggestions(&self.get_suggestions(&expr), query, start);
            }
        } else {
            let Some(last_dot) = input.rfind('.') else {
                return vec![];
            };
            let query = &input[..last_dot];
            let trailing = &input.get(last_dot + 1..).unwrap_or_default();
            let Ok(ast) = ast_parser::expr(query.trim()) else {
                return vec![];
            };
            if let Ok(expr) = self.parse(&ast) {
                let mut suggestions = self.get_suggestions(&expr);
                suggestions.retain(|s| s.contains(trailing));
                return apply_suggestions(&suggestions, query, start);
            }
        }

        vec![]
    }
}

pub(crate) fn parse_duration(duration_str: &str) -> Result<hifitime::Duration, Error> {
    let span = jiff::Span::from_str(duration_str)
        .map_err(|err| Error::InvalidMethodCall(err.to_string()))?;
    Ok(hifitime::Duration::from_nanoseconds(
        span.total(jiff::Unit::Nanosecond).unwrap(),
    ))
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("entity not found: {0}")]
    ComponentNotFound(String),
    #[error("invalid field access: {0}")]
    InvalidFieldAccess(String),
    #[error("invalid swizzle: {0}")]
    InvalidSwizzle(String),
    #[error("invalid method call: {0}")]
    InvalidMethodCall(String),
    #[error("parse {0}")]
    Parse(#[from] ParseError<peg::str::LineCol>),
    #[error("missing duration: {0}")]
    MissingDuration(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formulas::*;

    #[test]
    fn test_ast_parse() {
        assert_eq!(
            ast_parser::expr("test").unwrap(),
            AstNode::Ident(Cow::Borrowed("test"))
        );
        assert_eq!(
            ast_parser::expr("foo.bar").unwrap(),
            AstNode::Field(
                Box::new(AstNode::Ident(Cow::Borrowed("foo"))),
                Cow::Borrowed("bar")
            )
        );
        assert_eq!(
            ast_parser::expr("(a, b)").unwrap(),
            AstNode::Tuple(vec![
                AstNode::Ident(Cow::Borrowed("a")),
                AstNode::Ident(Cow::Borrowed("b")),
            ])
        );

        assert_eq!(
            ast_parser::expr("a.bar(b, c)").unwrap(),
            AstNode::MethodCall(
                Box::new(AstNode::Ident(Cow::Borrowed("a"))),
                Cow::Borrowed("bar"),
                vec![AstNode::Tuple(vec![
                    AstNode::Ident(Cow::Borrowed("b")),
                    AstNode::Ident(Cow::Borrowed("c")),
                ])]
            )
        )
    }

    fn create_test_entity_component() -> Arc<Component> {
        use impeller2::types::{ComponentId, PrimType};

        Arc::new(Component::new(
            "a.world_pos".to_string(),
            ComponentId::new("a.world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(), // 3D vector schema
        ))
    }

    fn create_test_component_part() -> Arc<ComponentPart> {
        Arc::new(ComponentPart {
            name: "a.world_pos".to_string(),
            id: ComponentId::new("a.world_pos"),
            component: Some(create_test_entity_component()),
            children: Default::default(),
        })
    }

    fn create_test_context() -> Context {
        Context::from_leaves(
            [create_test_entity_component()],
            Timestamp(0),    // earliest_timestamp
            Timestamp(1000), // last_timestamp
        )
    }

    #[test]
    fn test_component_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let expr = Expr::ComponentPart(Arc::new(ComponentPart {
            name: "a.world_pos".to_string(),
            id: ComponentId::new("a.world_pos"),
            component: Some(entity_component),
            children: Default::default(),
        }));
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos as 'a.world_pos' from a_world_pos"
        );
    }

    #[test]
    fn test_time_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let expr = Expr::Time(entity_component);
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select a_world_pos.time from a_world_pos");
    }

    #[test]
    fn test_fftfreq_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let time_expr = Expr::Time(entity_component);
        let expr = Expr::Formula(Arc::new(FftFreq), Box::new(time_expr));
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select fftfreq(a_world_pos.time) from a_world_pos"
        );
    }

    #[test]
    fn test_fft_sql_via_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.world_pos[0].fft()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select fft(a_world_pos.a_world_pos[1]) as 'fft(a.world_pos.x)' from a_world_pos"
        );
    }

    #[test]
    fn test_fftfreq_sql_via_parse() {
        let context = create_test_context();
        let expr = context.parse_str("a.world_pos.time.fftfreq()").unwrap();
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(sql, "select fftfreq(a_world_pos.time) from a_world_pos");
    }

    #[test]
    fn test_array_access_sql() {
        let part = create_test_component_part();
        let context = create_test_context();

        // Test first element
        let expr = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 0);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos[1] as 'a.world_pos.x' from a_world_pos"
        );

        // Test second element
        let expr = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 1);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos[2] as 'a.world_pos.y' from a_world_pos"
        );

        // Test third element
        let expr = Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 2);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos[3] as 'a.world_pos.z' from a_world_pos"
        );
    }

    #[test]
    fn test_single_table_tuple_sql() {
        let part = create_test_component_part();
        let context = create_test_context();

        // Test Tuple with Time and ArrayAccess
        let expr = Expr::Tuple(vec![
            Expr::Time(part.component.clone().unwrap()),
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 0),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, a_world_pos.a_world_pos[1] as 'a.world_pos.x' from a_world_pos"
        );

        // Test Tuple with multiple ArrayAccess
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 1),
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part.clone())), 2),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos[1] as 'a.world_pos.x', a_world_pos.a_world_pos[2] as 'a.world_pos.y', a_world_pos.a_world_pos[3] as 'a.world_pos.z' from a_world_pos"
        );
    }

    #[test]
    fn test_two_table_join_sql() {
        use impeller2::types::{ComponentId, PrimType};

        let part1 = create_test_component_part();
        let context = create_test_context();

        let component2 = Arc::new(Component::new(
            "b.velocity".to_string(),
            ComponentId::new("b.velocity"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        let part2 = Arc::new(ComponentPart {
            name: "b.velocity".to_string(),
            id: ComponentId::new("b.velocity"),
            component: Some(component2),
            children: BTreeMap::default(),
        });

        // Test Tuple with components from different tables
        let expr = Expr::Tuple(vec![
            Expr::ComponentPart(part1.clone()),
            Expr::ComponentPart(part2.clone()),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos as 'a.world_pos', b_velocity.b_velocity as 'b.velocity' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with Time from one table and Component from another
        let expr = Expr::Tuple(vec![
            Expr::Time(part1.component.clone().unwrap()),
            Expr::ComponentPart(part2.clone()),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, b_velocity.b_velocity as 'b.velocity' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with ArrayAccess from different tables
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part1.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::ComponentPart(part2.clone())), 1),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.a_world_pos[1] as 'a.world_pos.x', b_velocity.b_velocity[2] as 'b.velocity.y' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );
    }

    #[test]
    fn test_three_table_join_sql() {
        use impeller2::types::{ComponentId, PrimType};

        let part1 = create_test_component_part();
        let context = create_test_context();

        let component2 = Arc::new(Component::new(
            "b.velocity".to_string(),
            ComponentId::new("b.velocity"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        let part2 = Arc::new(ComponentPart {
            name: "b.velocity".to_string(),
            id: ComponentId::new("b.velocity"),
            component: Some(component2),
            children: BTreeMap::default(),
        });

        let component3 = Arc::new(Component::new(
            "c.acceleration".to_string(),
            ComponentId::new("c.acceleration"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        let part3 = Arc::new(ComponentPart {
            name: "c.acceleration".to_string(),
            id: ComponentId::new("c.acceleration"),
            component: Some(component3),
            children: BTreeMap::default(),
        });

        let expr = Expr::Tuple(vec![
            Expr::ComponentPart(part1.clone()),
            Expr::ComponentPart(part2.clone()),
            Expr::ComponentPart(part3.clone()),
        ]);
        let result = expr.to_sql(&context);
        let result_str = result.unwrap();
        assert_eq!(
            result_str,
            "select a_world_pos.a_world_pos as 'a.world_pos', b_velocity.b_velocity as 'b.velocity', c_acceleration.c_acceleration as 'c.acceleration' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time JOIN c_acceleration ON a_world_pos.time = c_acceleration.time"
        );
    }

    #[test]
    fn test_norm_sql() {
        // norm()
        let part = create_test_component_part();
        let context = create_test_context();
        let expr = Expr::Formula(Arc::new(Norm), Box::new(Expr::ComponentPart(part)));
        let sql = expr.to_sql(&context).unwrap();
        assert_eq!(
            sql,
            "select sqrt(a_world_pos.a_world_pos[1] * a_world_pos.a_world_pos[1] + a_world_pos.a_world_pos[2] * a_world_pos.a_world_pos[2] + a_world_pos.a_world_pos[3] * a_world_pos.a_world_pos[3]) as 'norm(a.world_pos)' from a_world_pos"
        );
    }

    #[test]
    fn test_element_names() {
        assert_eq!(default_element_names(&[4]), vec!["x", "y", "z", "w"]);
        assert_eq!(default_element_names(&[2, 2]), vec!["xx", "xy", "yx", "yy"]);
    }

    #[test]
    fn test_entity_suggestions() {
        let context = create_test_context();
        let part = context.component_parts.get("a").unwrap();
        let expr = Expr::ComponentPart(Arc::new(part.clone()));

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"world_pos".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
        // New formulas: degrees(), sqrt() are also suggested for ComponentPart
        assert!(suggestions.len() >= 4);
    }

    #[test]
    fn test_component_suggestions() {
        let context = create_test_context();
        let entity = context.component_parts.get("a").unwrap();
        let component = entity.children.get("world_pos").unwrap();
        let expr = Expr::ComponentPart(Arc::new(component.clone()));

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
        assert!(suggestions.contains(&"norm()".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert!(suggestions.contains(&"x".to_string()));
        assert!(suggestions.contains(&"y".to_string()));
        assert!(suggestions.contains(&"z".to_string()));
        assert!(!suggestions.contains(&"w".to_string()));
    }

    #[test]
    fn test_time_suggestions() {
        let component = create_test_entity_component();
        let expr = Expr::Time(component);
        let context = create_test_context();

        let suggestions = context.get_suggestions(&expr);
        assert_eq!(suggestions, vec!["fftfreq()"]);
    }

    #[test]
    fn test_array_access_suggestions() {
        let component = create_test_component_part();
        let expr = Expr::ArrayAccess(Box::new(Expr::ComponentPart(component)), 0);
        let context = create_test_context();

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"fft()".to_string()));
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
    }

    #[test]
    fn test_formula_suggestions() {
        let context = create_test_context();
        let expr = context.parse_str("a.world_pos[0].fft()").unwrap();
        let suggestions = context.get_suggestions(&expr);
        // After fft(), we can chain degrees() and sqrt() as well
        assert!(suggestions.contains(&"first".to_string()));
        assert!(suggestions.contains(&"last".to_string()));
        assert!(suggestions.contains(&"degrees()".to_string()));
        assert!(suggestions.contains(&"sqrt()".to_string()));
    }

    #[test]
    fn test_string_suggestions_empty() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"a".to_string()));
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_string_suggestions_with_period() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("a.")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"world_pos".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
    }

    #[test]
    fn test_string_suggestions_component_with_period() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("a.world_pos.")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
        assert!(suggestions.contains(&"norm()".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert!(suggestions.contains(&"x".to_string()));
    }

    #[test]
    fn test_string_suggestions_invalid_input() {
        let context = create_test_context();
        let suggestions = context.get_string_suggestions("invalid_entity.");

        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_trailing_int_ident() {
        assert_eq!(
            ast_parser::expr("cow_0").unwrap(),
            AstNode::Ident(Cow::Borrowed("cow_0"))
        );
    }

    #[test]
    fn test_fmt_string() {
        assert_eq!(
            ast_parser::fmt_string("test${foo}test").unwrap(),
            vec![
                FmtNode::String(Cow::Borrowed("test")),
                FmtNode::AstNode(AstNode::Ident(Cow::Borrowed("foo"))),
                FmtNode::String(Cow::Borrowed("test")),
            ]
        );
        assert_eq!(
            ast_parser::fmt_string("test${x + 1}test").unwrap(),
            vec![
                FmtNode::String(Cow::Borrowed("test")),
                FmtNode::AstNode(AstNode::BinaryOp(
                    Box::new(AstNode::Ident(Cow::Borrowed("x"))),
                    Box::new(AstNode::FloatLiteral(1.0)),
                    BinaryOp::Add
                )),
                FmtNode::String(Cow::Borrowed("test")),
            ]
        );
    }

    #[test]
    fn test_rocket_string() {
        assert_eq!(
            ast_parser::expr("rocket.fin_deflect[0]").unwrap(),
            AstNode::ArrayIndex(
                Box::new(AstNode::Field(
                    Box::new(AstNode::Ident("rocket".into())),
                    "fin_deflect".into()
                )),
                0
            )
        );
    }

    // Integration tests for complex formula combinations
    mod integration_tests {
        use super::*;

        fn create_rocket_context() -> Context {
            // Create a 3D velocity vector component like rocket.v_body
            let v_body_comp = Arc::new(Component::new(
                "rocket.v_body".to_string(),
                ComponentId::new("rocket.v_body"),
                Schema::new(impeller2::types::PrimType::F64, vec![3u64]).unwrap(),
            ));
            Context::from_leaves([v_body_comp], Timestamp(0), Timestamp(1000))
        }

        #[test]
        fn test_angle_of_attack_calculation() {
            // Test the rocket angle-of-attack calculation pattern:
            // ((rocket.v_body[0] * -1.0) / rocket.v_body.norm().clip(min, max)).arccos().degrees() * (rocket.v_body[2] * -1.0).sign()
            let context = create_rocket_context();

            let expr = context
                .parse_str(
                    "((rocket.v_body[0] * -1.0) / rocket.v_body.norm().clip(0.000000001, 999999)).arccos().degrees() * (rocket.v_body[2] * -1.0).sign()",
                )
                .unwrap();

            let sql = expr.to_sql(&context).unwrap();

            // Verify all formulas are present
            assert!(sql.contains("degrees("), "Should contain degrees()");
            assert!(sql.contains("acos("), "Should contain acos() from arccos()");
            assert!(sql.contains("sqrt("), "Should contain sqrt() from norm()");
            assert!(
                sql.contains("GREATEST("),
                "Should contain GREATEST from clip()"
            );
            assert!(sql.contains("LEAST("), "Should contain LEAST from clip()");
            assert!(sql.contains("CASE WHEN"), "Should contain CASE from sign()");
            assert!(
                sql.contains("rocket_v_body"),
                "Should reference rocket.v_body table"
            );

            // Verify the structure is correct (degrees wraps acos)
            assert!(
                sql.contains("degrees(acos("),
                "degrees() should wrap acos()"
            );
        }

        #[test]
        fn test_complex_chained_formulas() {
            // Test: (value).sqrt().abs().degrees()
            let component = Arc::new(Component::new(
                "a.value".to_string(),
                ComponentId::new("a.value"),
                Schema::new(impeller2::types::PrimType::F64, Vec::<u64>::new()).unwrap(),
            ));
            let context = Context::from_leaves([component], Timestamp(0), Timestamp(1000));

            let expr = context.parse_str("a.value.sqrt().abs().degrees()").unwrap();
            let sql = expr.to_sql(&context).unwrap();

            // Verify chaining order: degrees(abs(sqrt(value)))
            assert!(sql.contains("sqrt("), "Should contain sqrt()");
            assert!(sql.contains("abs("), "Should contain abs()");
            assert!(sql.contains("degrees("), "Should contain degrees()");

            // Verify proper nesting
            assert!(
                sql.contains("degrees(abs(sqrt("),
                "Formulas should be properly nested"
            );
        }

        #[test]
        fn test_normalized_vector_with_clip() {
            // Test: (v_body[0] / v_body.norm().clip(min, max))
            // Note: EQL doesn't support scientific notation, so use full decimal
            let context = create_rocket_context();

            let expr = context
                .parse_str("rocket.v_body[0] / rocket.v_body.norm().clip(0.000000000001, 999999)")
                .unwrap();

            let sql = expr.to_sql(&context).unwrap();

            // Verify components
            assert!(sql.contains("sqrt("), "norm() should generate sqrt()");
            assert!(sql.contains("GREATEST("), "clip() should generate GREATEST");
            assert!(sql.contains("LEAST("), "clip() should generate LEAST");
            assert!(sql.contains(" / "), "Should contain division operator");
        }

        #[test]
        fn test_atan2_degrees_pattern() {
            // Test common pattern: y.atan2(x).degrees()
            let y_comp = Arc::new(Component::new(
                "a.y".to_string(),
                ComponentId::new("a.y"),
                Schema::new(impeller2::types::PrimType::F64, Vec::<u64>::new()).unwrap(),
            ));
            let x_comp = Arc::new(Component::new(
                "a.x".to_string(),
                ComponentId::new("a.x"),
                Schema::new(impeller2::types::PrimType::F64, Vec::<u64>::new()).unwrap(),
            ));
            let context = Context::from_leaves([y_comp, x_comp], Timestamp(0), Timestamp(1000));

            let expr = context.parse_str("a.y.atan2(a.x).degrees()").unwrap();
            let sql = expr.to_sql(&context).unwrap();

            // Verify structure
            assert!(
                sql.contains("degrees(atan2("),
                "degrees() should wrap atan2()"
            );
            // Both components should be referenced
            assert!(sql.contains("a_y"), "Should reference a.y component");
            assert!(sql.contains("a_x"), "Should reference a.x component");
        }

        #[test]
        fn test_sign_with_multiplication() {
            // Test: (value * -1.0).sign()
            let component = Arc::new(Component::new(
                "a.value".to_string(),
                ComponentId::new("a.value"),
                Schema::new(impeller2::types::PrimType::F64, Vec::<u64>::new()).unwrap(),
            ));
            let context = Context::from_leaves([component], Timestamp(0), Timestamp(1000));

            let expr = context.parse_str("(a.value * -1.0).sign()").unwrap();
            let sql = expr.to_sql(&context).unwrap();

            // Verify CASE statement structure
            assert!(sql.contains("CASE WHEN"), "Should contain CASE statement");
            assert!(sql.contains(" * -1"), "Should contain multiplication by -1");
        }

        #[test]
        fn test_arccos_clipping_integration() {
            // Test that arccos automatically clips input to [-1, 1]
            let component = Arc::new(Component::new(
                "a.value".to_string(),
                ComponentId::new("a.value"),
                Schema::new(impeller2::types::PrimType::F64, Vec::<u64>::new()).unwrap(),
            ));
            let context = Context::from_leaves([component], Timestamp(0), Timestamp(1000));

            let expr = context
                .parse_str("(a.value / 100.0).arccos().degrees()")
                .unwrap();
            let sql = expr.to_sql(&context).unwrap();

            // Verify arccos includes clipping
            assert!(
                sql.contains("GREATEST(-1.0, LEAST(1.0"),
                "arccos should clip input to [-1, 1]"
            );
            assert!(sql.contains("acos("), "Should contain acos function");
            assert!(sql.contains("degrees("), "Should contain degrees function");
        }

        #[test]
        fn test_angle_of_attack_sql_structure() {
            // Diagnostic test for the rocket angle-of-attack calculation
            // This test documents the exact SQL structure generated
            let context = create_rocket_context();

            let expr = context
                .parse_str(
                    "((rocket.v_body[0] * -1.0) / rocket.v_body.norm().clip(0.000000001, 999999)).arccos().degrees() * (rocket.v_body[2] * -1.0).sign()",
                )
                .unwrap();

            let sql = expr.to_sql(&context).unwrap();

            // Print SQL for debugging (will show in test output if run with --nocapture)
            eprintln!("Generated SQL for angle-of-attack:");
            eprintln!("{}", sql);

            // Verify the mathematical structure:
            // degrees(acos(GREATEST(-1, LEAST(1, -v[0] / GREATEST(min, LEAST(norm, max)))))) * CASE...

            // Check for correct nesting of functions
            assert!(
                sql.contains("degrees("),
                "Outermost function should be degrees()"
            );
            assert!(sql.contains("acos("), "Should use acos() for arccos");
            assert!(sql.contains("sqrt("), "norm() should use sqrt()");
            assert!(
                sql.contains("CASE WHEN"),
                "sign() should use CASE statement"
            );

            // Verify the clipping is present
            assert!(
                sql.contains("GREATEST(-1.0, LEAST(1.0"),
                "arccos should clip to [-1, 1]"
            );
            assert!(
                sql.contains("GREATEST(0.000000001"),
                "Should clip norm to min value"
            );

            // Verify multiplication operator is present for final sign multiplication
            assert!(
                sql.contains(" * CASE"),
                "Should multiply arccos result by sign"
            );

            // Note: If the angle-of-attack shows a +90° offset in the actual application,
            // possible causes could be:
            // 1. Coordinate system mismatch (the Python code uses thrust_vector = [-1,0,0])
            // 2. The arccos clipping behavior differences between Python and SQL
            // 3. Sign function behavior with zero values
            //
            // To debug further:
            // - Run: cargo test -p eql --lib test_angle_of_attack_sql_structure -- --nocapture
            // - This will print the generated SQL
            // - Compare individual component values (v_body[0], norm, sign) in the database
            //   vs Python to identify where the discrepancy occurs
        }
    }
}
