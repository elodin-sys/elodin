use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    str::FromStr,
    sync::Arc,
};
use unicode_ident::*;

use impeller2::{
    schema::Schema,
    types::{ComponentId, EntityId, Timestamp},
};
use peg::error::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode<'input> {
    Ident(Cow<'input, str>),
    Field(Box<AstNode<'input>>, Cow<'input, str>),
    MethodCall(Box<AstNode<'input>>, Cow<'input, str>, Vec<AstNode<'input>>),
    BinaryOp(Box<AstNode<'input>>, Box<AstNode<'input>>, BinaryOp),
    Tuple(Vec<AstNode<'input>>),
    StringLiteral(Cow<'input, str>),
    FloatLiteral(f64),
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
        rule num() -> f64
            = s:$("-"? ['0'..='9']+ ("." ['0'..='9']*)?) { s.parse().unwrap() }

        rule string_literal() -> Cow<'input, str> = "\"" s:$([^'"']*) "\"" { Cow::Borrowed(s) }
        rule comma() = ("," _?)
        rule binary_op() -> BinaryOp = "+" { BinaryOp::Add } / "-" { BinaryOp::Sub } / "*" { BinaryOp::Mul } / "/" { BinaryOp::Div }

        pub rule expr() -> AstNode<'input> = precedence! {
        a:(@) comma() b:@ { AstNode::Tuple(vec![a, b]) }
        --
        a:(@) _ op:binary_op() _ b:@ { AstNode::BinaryOp(Box::new(a), Box::new(b), op) }
        --
        e:(@) "." i:ident_str() "(" args:expr() ** comma() ")" { AstNode::MethodCall(Box::new(e), i,  args) }
        --
        e:(@) "." i:ident_str() { AstNode::Field(Box::new(e), i) }
        --
        "(" _ e:expr() _ ")" { e }
        --
        f:num() { AstNode::FloatLiteral(f) }
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
    Entity(Arc<Entity>),
    Component(Arc<EntityComponent>),
    Time(Arc<EntityComponent>),
    ArrayAccess(Box<Expr>, usize),
    Tuple(Vec<Expr>),
    DurationLiteral(hifitime::Duration),
    FloatLiteral(f64),

    // ffts
    Fft(Box<Expr>),
    FftFreq(Box<Expr>),

    // time limits
    Last(Box<Expr>, hifitime::Duration),
    First(Box<Expr>, hifitime::Duration),

    // infix
    BinaryOp(Box<Expr>, Box<Expr>, BinaryOp),
}

impl Expr {
    fn to_field(&self) -> Result<String, Error> {
        match self {
            Expr::Component(component) => Ok(component.name.clone()),

            Expr::Time(_) => Ok("time".to_string()),
            Expr::Fft(e) => Ok(format!("fft({})", e.to_field()?)),
            Expr::FftFreq(e) => Ok(format!("fftfreq({})", e.to_field()?)),
            Expr::BinaryOp(left, right, op) => Ok(format!(
                "({} {} {})",
                left.to_field()?,
                op.to_str(),
                right.to_field()?
            )),

            Expr::ArrayAccess(inner_expr, index) => match inner_expr.as_ref() {
                Expr::Component(component) => Ok(format!("{}[{}]", component.name, index + 1)),
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
            Expr::Component(component) => {
                Ok(format!("{}_{}", component.entity_name, component.name))
            }

            Expr::Time(component) => Ok(format!("{}_{}", component.entity_name, component.name)),
            Expr::FftFreq(e) => e.to_table(),
            Expr::Fft(e) => e.to_table(),
            Expr::BinaryOp(left, _, _) => left.to_table(),

            Expr::ArrayAccess(inner_expr, _) => match inner_expr.as_ref() {
                Expr::Component(component) => {
                    Ok(format!("{}_{}", component.entity_name, component.name))
                }
                _ => Err(Error::InvalidFieldAccess(
                    "array access on non-component".to_string(),
                )),
            },

            expr => Err(Error::InvalidFieldAccess(format!(
                "unsupported expression type for table {expr:?}"
            ))),
        }
    }

    /// Converts an Expr to a qualified SQL field name (table.field) for use in JOINs.
    fn to_qualified_field(&self) -> Result<String, Error> {
        match self {
            Expr::Fft(e) => Ok(format!("fft({})", e.to_qualified_field()?)),
            Expr::FftFreq(e) => Ok(format!("fftfreq({})", e.to_qualified_field()?)),
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
            Expr::Fft(e) => Some(format!("fft({})", e.to_column_name()?)),
            Expr::FftFreq(e) => Some(format!("fftfreq({})", e.to_column_name()?)),
            Expr::Component(e) => Some(format!("{}.{}", e.entity_name, e.name)),
            Expr::ArrayAccess(expr, index) => match expr.as_ref() {
                Expr::Component(c) => c
                    .element_names
                    .get(*index)
                    .map(|name| format!("{}.{}.{}", c.entity_name, c.name, name)),
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

    fn to_sql_time_field(&self) -> Result<String, Error> {
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
            Expr::Entity(_entity) => {
                // For bare entities, just return a placeholder or error
                // This matches the AstNode::Ident behavior which just returns the name
                Err(Error::InvalidFieldAccess(
                    "cannot convert bare entity to SQL".to_string(),
                ))
            }

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

            Expr::Last(expr, duration) => {
                let sql = expr.to_sql(context)?;
                let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
                let lower_bound = context.last_timestamp.0 - duration_micros;
                let lower_bound = lower_bound as f64 * 1e-6;
                let time_field = expr.to_sql_time_field()?;
                Ok(format!(
                    "{sql} where {time_field} >= to_timestamp({lower_bound})",
                ))
            }
            Expr::First(expr, duration) => {
                let sql = expr.to_sql(context)?;
                let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
                let upper_bound = context.earliest_timestamp.0 + duration_micros;
                let upper_bound = upper_bound as f64 * 1e-6;
                let time_field = expr.to_sql_time_field()?;
                Ok(format!(
                    "{sql} where {time_field} <= to_timestamp({upper_bound})",
                ))
            }

            Expr::DurationLiteral(_) => Err(Error::InvalidFieldAccess(
                "cannot convert duration literal to SQL".to_string(),
            )),

            expr => Ok(format!(
                "select {} from {}",
                expr.to_select_part()?,
                expr.to_table()?
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub name: String,
    pub components: HashMap<String, Arc<EntityComponent>>,
}

#[derive(Clone, Debug)]
pub struct EntityComponent {
    pub name: String,
    pub entity_name: String,
    pub entity: EntityId,
    pub id: ComponentId,
    pub schema: Schema,
    pub element_names: Vec<String>,
}

impl EntityComponent {
    pub fn new(
        name: String,
        entity_name: String,
        entity: EntityId,
        id: ComponentId,
        schema: Schema,
    ) -> Self {
        let element_names = default_element_names(schema.dim());
        Self {
            name,
            entity_name,
            entity,
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

pub struct Context {
    pub entities: HashMap<String, Arc<Entity>>,
    pub earliest_timestamp: Timestamp,
    pub last_timestamp: Timestamp,
}

impl Context {
    pub fn new(
        entities: HashMap<String, Arc<Entity>>,
        earliest_timestamp: Timestamp,
        last_timestamp: Timestamp,
    ) -> Self {
        Self {
            entities,
            earliest_timestamp,
            last_timestamp,
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
                .entities
                .get(cow.as_ref())
                .ok_or(Error::EntityNotFound(cow.to_string()))
                .cloned()
                .map(Expr::Entity),
            AstNode::Field(ast_node, cow) => {
                let expr = self.parse(ast_node)?;
                match &expr {
                    Expr::Entity(entity) => entity
                        .components
                        .get(cow.as_ref())
                        .ok_or(Error::ComponentNotFound(cow.to_string()))
                        .cloned()
                        .map(Expr::Component),
                    Expr::Component(c) => match cow.as_ref() {
                        "time" => Ok(Expr::Time(c.clone())),
                        _ => {
                            let offset = c
                                .element_names
                                .iter()
                                .position(|name| name == cow.as_ref())
                                .ok_or(Error::InvalidSwizzle(cow.to_string()))?;
                            Ok(Expr::ArrayAccess(Box::new(expr), offset))
                        }
                    },
                    _ => Err(Error::InvalidFieldAccess(cow.to_string())),
                }
            }
            AstNode::MethodCall(recv, cow, ast_nodes) => {
                let recv = self.parse(recv)?;
                let args = ast_nodes
                    .iter()
                    .map(|ast_node| self.parse(ast_node))
                    .collect::<Result<Vec<_>, _>>()?;
                match (cow.as_ref(), &recv, &args[..]) {
                    ("fft", Expr::ArrayAccess(_, _), &[]) => Ok(Expr::Fft(Box::new(recv))),
                    ("fftfreq", Expr::Time(_), &[]) => Ok(Expr::FftFreq(Box::new(recv))),
                    ("last", _, &[Expr::DurationLiteral(duration)]) => {
                        Ok(Expr::Last(Box::new(recv), duration))
                    }
                    ("first", _, &[Expr::DurationLiteral(duration)]) => {
                        Ok(Expr::First(Box::new(recv), duration))
                    }
                    _ => Err(Error::InvalidMethodCall(cow.to_string())),
                }
            }
            AstNode::Tuple(ast_nodes) => ast_nodes
                .iter()
                .map(|ast_node| self.parse(ast_node))
                .collect::<Result<Vec<_>, _>>()
                .map(Expr::Tuple),
            AstNode::StringLiteral(s) => {
                // Parse duration strings like "5m", "10s", "1h"
                self.parse_duration(s.as_ref()).map(Expr::DurationLiteral)
            }
            AstNode::BinaryOp(left, right, op) => {
                let left = self.parse(left)?;
                let right = self.parse(right)?;
                Ok(Expr::BinaryOp(Box::new(left), Box::new(right), *op))
            }
            AstNode::FloatLiteral(f) => Ok(Expr::FloatLiteral(*f)),
        }
    }

    fn parse_duration(&self, duration_str: &str) -> Result<hifitime::Duration, Error> {
        let span = jiff::Span::from_str(duration_str)
            .map_err(|err| Error::InvalidMethodCall(err.to_string()))?;
        Ok(hifitime::Duration::from_nanoseconds(
            span.total(jiff::Unit::Nanosecond).unwrap(),
        ))
    }

    /// Get suggestions for the given expression
    pub fn get_suggestions(&self, expr: &Expr) -> Vec<String> {
        match expr {
            Expr::Entity(entity) => {
                let mut suggestions = entity.components.keys().cloned().collect::<Vec<_>>();
                suggestions.sort();
                suggestions.push("time".to_string());
                suggestions
            }
            Expr::Component(component) => component
                .element_names
                .iter()
                .cloned()
                .chain([
                    "last(".to_string(),
                    "first(".to_string(),
                    "time".to_string(),
                ])
                .collect(),
            Expr::Time(_) => {
                vec!["fftfreq()".to_string()]
            }
            Expr::ArrayAccess(_, _) => {
                vec![
                    "fft()".to_string(),
                    "last(".to_string(),
                    "first(".to_string(),
                ]
            }
            Expr::Tuple(_) => {
                vec!["last".to_string(), "first".to_string()]
            }
            Expr::DurationLiteral(_) => {
                vec![]
            }
            Expr::Fft(_) => {
                vec!["last".to_string(), "first".to_string()]
            }
            Expr::FftFreq(_) => {
                vec!["last".to_string(), "first".to_string()]
            }
            _ => vec![],
        }
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
            let mut entity_names = self.entities.keys().cloned().collect::<Vec<_>>();
            entity_names.sort();
            return entity_names
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("entity not found: {0}")]
    EntityNotFound(String),
    #[error("component not found: {0}")]
    ComponentNotFound(String),
    #[error("invalid field access: {0}")]
    InvalidFieldAccess(String),
    #[error("invalid swizzle: {0}")]
    InvalidSwizzle(String),
    #[error("invalid method call: {0}")]
    InvalidMethodCall(String),
    #[error("parse {0}")]
    Parse(#[from] ParseError<peg::str::LineCol>),
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn create_test_entity_component() -> Arc<EntityComponent> {
        use impeller2::types::{ComponentId, EntityId, PrimType};

        Arc::new(EntityComponent::new(
            "world_pos".to_string(),
            "a".to_string(),
            EntityId(1),
            ComponentId::new("world_pos"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(), // 3D vector schema
        ))
    }

    fn create_test_context() -> Context {
        let mut entities = HashMap::new();

        let component = create_test_entity_component();
        let mut components = HashMap::new();
        components.insert("component".to_string(), component);

        let entity = Arc::new(Entity {
            name: "test_entity".to_string(),
            components,
        });

        entities.insert("test_entity".to_string(), entity);

        Context::new(
            entities,
            Timestamp(0),    // earliest_timestamp
            Timestamp(1000), // last_timestamp
        )
    }

    #[test]
    fn test_component_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let expr = Expr::Component(entity_component);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos as 'a.world_pos' from a_world_pos"
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
        let expr = Expr::FftFreq(Box::new(time_expr));
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select fftfreq(a_world_pos.time) from a_world_pos"
        );
    }

    #[test]
    fn test_array_access_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        // Test first element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[1] as 'a.world_pos.x' from a_world_pos"
        );

        // Test second element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 1);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[2] as 'a.world_pos.y' from a_world_pos"
        );

        // Test third element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 2);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[3] as 'a.world_pos.z' from a_world_pos"
        );
    }

    #[test]
    fn test_single_table_tuple_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        // Test Tuple with Time and ArrayAccess
        let expr = Expr::Tuple(vec![
            Expr::Time(entity_component.clone()),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, a_world_pos.world_pos[1] as 'a.world_pos.x' from a_world_pos"
        );

        // Test Tuple with multiple ArrayAccess
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 1),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 2),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[1] as 'a.world_pos.x', a_world_pos.world_pos[2] as 'a.world_pos.y', a_world_pos.world_pos[3] as 'a.world_pos.z' from a_world_pos"
        );
    }

    #[test]
    fn test_two_table_join_sql() {
        use impeller2::types::{ComponentId, EntityId, PrimType};

        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let entity_component2 = Arc::new(EntityComponent::new(
            "velocity".to_string(),
            "b".to_string(),
            EntityId(2),
            ComponentId::new("velocity"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        // Test Tuple with components from different tables
        let expr = Expr::Tuple(vec![
            Expr::Component(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos as 'a.world_pos', b_velocity.velocity as 'b.velocity' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with Time from one table and Component from another
        let expr = Expr::Tuple(vec![
            Expr::Time(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, b_velocity.velocity as 'b.velocity' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with ArrayAccess from different tables
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component2.clone())), 1),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[1] as 'a.world_pos.x', b_velocity.velocity[2] as 'b.velocity.y' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );
    }

    #[test]
    fn test_three_table_join_sql() {
        use impeller2::types::{ComponentId, EntityId, PrimType};

        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let entity_component2 = Arc::new(EntityComponent::new(
            "velocity".to_string(),
            "b".to_string(),
            EntityId(2),
            ComponentId::new("velocity"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        let entity_component3 = Arc::new(EntityComponent::new(
            "acceleration".to_string(),
            "c".to_string(),
            EntityId(3),
            ComponentId::new("acceleration"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        let expr = Expr::Tuple(vec![
            Expr::Component(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
            Expr::Component(entity_component3.clone()),
        ]);
        let result = expr.to_sql(&context);
        let result_str = result.unwrap();
        assert_eq!(
            result_str,
            "select a_world_pos.world_pos as 'a.world_pos', b_velocity.velocity as 'b.velocity', c_acceleration.acceleration as 'c.acceleration' from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time JOIN c_acceleration ON a_world_pos.time = c_acceleration.time"
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
        let entity = context.entities.get("test_entity").unwrap();
        let expr = Expr::Entity(entity.clone());

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"component".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert_eq!(suggestions.len(), 2);
    }

    #[test]
    fn test_component_suggestions() {
        let context = create_test_context();
        let entity = context.entities.get("test_entity").unwrap();
        let component = entity.components.get("component").unwrap();
        let expr = Expr::Component(component.clone());

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
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
        let component = create_test_entity_component();
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(component)), 0);
        let context = create_test_context();

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"fft()".to_string()));
        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
    }

    #[test]
    fn test_string_suggestions_empty() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"test_entity".to_string()));
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_string_suggestions_with_period() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("test_entity.")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"component".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
    }

    #[test]
    fn test_string_suggestions_component_with_period() {
        let context = create_test_context();
        let suggestions = context
            .get_string_suggestions("test_entity.component.")
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();

        assert!(suggestions.contains(&"first(".to_string()));
        assert!(suggestions.contains(&"last(".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
        assert!(suggestions.contains(&"x".to_string()));
    }

    #[test]
    fn test_string_suggestions_invalid_input() {
        let context = create_test_context();
        let suggestions = context.get_string_suggestions("invalid_entity.");

        assert!(suggestions.is_empty());
    }
}
