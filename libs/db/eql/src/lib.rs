use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    str::FromStr,
    sync::Arc,
};

use impeller2::{
    schema::Schema,
    types::{ComponentId, EntityId, Timestamp},
};
use peg::error::ParseError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstNode<'input> {
    Ident(Cow<'input, str>),
    Field(Box<AstNode<'input>>, Cow<'input, str>),
    MethodCall(Box<AstNode<'input>>, Cow<'input, str>, Vec<AstNode<'input>>),
    Tuple(Vec<AstNode<'input>>),
    StringLiteral(Cow<'input, str>),
}

peg::parser! {
    grammar ast_parser() for str {
        rule _ = quiet!{[' ' | '\n' | '\t']*}
        rule ident_str() -> Cow<'input, str> = s:$(['a'..='z' | 'A'..='Z' | '0'..='9' | '_']+) { Cow::Borrowed(s) }
        rule string_literal() -> Cow<'input, str> = "\"" s:$([^'"']*) "\"" { Cow::Borrowed(s) }
        rule comma() = ("," _?)

        pub rule expr() -> AstNode<'input> = precedence! {
        e:(@) "." i:ident_str() "(" args:expr() ** comma() ")" { AstNode::MethodCall(Box::new(e), i,  args) }
        --
        "(" e:expr() ** comma() ")" { AstNode::Tuple(e) }
        --
        e:(@) "." i:ident_str() { AstNode::Field(Box::new(e), i) }
        --
        s:string_literal() { AstNode::StringLiteral(s) }
        --
        s:ident_str() { AstNode::Ident(s) }
        }
    }
}

pub enum Expr {
    // core
    Entity(Arc<Entity>),
    Component(Arc<EntityComponent>),
    Time(Arc<EntityComponent>),
    ArrayAccess(Box<Expr>, usize),
    Tuple(Vec<Expr>),
    DurationLiteral(hifitime::Duration),

    // ffts
    Fft(Box<Expr>),
    FftFreq(Box<Expr>),

    // time limits
    Last(Box<Expr>, hifitime::Duration),
    First(Box<Expr>, hifitime::Duration),
}

impl Expr {
    fn to_field(&self) -> Result<String, Error> {
        match self {
            Expr::Component(component) => Ok(component.name.clone()),

            Expr::Time(_) => Ok("time".to_string()),
            Expr::Fft(e) => Ok(format!("fft({})", e.to_field()?)),
            Expr::FftFreq(e) => Ok(format!("fftfreq({})", e.to_field()?)),

            Expr::ArrayAccess(inner_expr, index) => match inner_expr.as_ref() {
                Expr::Component(component) => Ok(format!("{}[{}]", component.name, index + 1)),
                _ => Err(Error::InvalidFieldAccess(
                    "array access on non-component".to_string(),
                )),
            },

            _ => Err(Error::InvalidFieldAccess(
                "unsupported expression type for field".to_string(),
            )),
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

            Expr::ArrayAccess(inner_expr, _) => match inner_expr.as_ref() {
                Expr::Component(component) => {
                    Ok(format!("{}_{}", component.entity_name, component.name))
                }
                _ => Err(Error::InvalidFieldAccess(
                    "array access on non-component".to_string(),
                )),
            },

            _ => Err(Error::InvalidFieldAccess(
                "unsupported expression type for table".to_string(),
            )),
        }
    }

    /// Converts an Expr to a qualified SQL field name (table.field) for use in JOINs.
    fn to_qualified_field(&self) -> Result<String, Error> {
        match self {
            Expr::Fft(e) => Ok(format!("fft({})", e.to_qualified_field()?)),
            Expr::FftFreq(e) => Ok(format!("fftfreq({})", e.to_qualified_field()?)),
            _ => {
                let table = self.to_table()?;
                let field = self.to_field()?;
                Ok(format!("{}.{}", table, field))
            }
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
                        select_parts.push(element.to_field()?);
                    }
                    Ok(format!(
                        "select {} from {}",
                        select_parts.join(", "),
                        table_names.first().unwrap()
                    ))
                } else {
                    let mut select_parts = Vec::new();
                    for element in elements {
                        select_parts.push(element.to_qualified_field()?);
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

            Expr::Last(expr, duration) => {
                let sql = expr.to_sql(&context)?;
                let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
                let lower_bound = context.last_timestamp.0 - duration_micros;
                let lower_bound = lower_bound as f64 * 1e-6;
                Ok(format!(
                    "{} where time >= to_timestamp({})",
                    sql, lower_bound
                ))
            }
            Expr::First(expr, duration) => {
                let sql = expr.to_sql(&context)?;
                let duration_micros = (duration.total_nanoseconds() / 1000) as i64;
                let upper_bound = context.earliest_timestamp.0 + duration_micros;
                let upper_bound = upper_bound as f64 * 1e-6;
                Ok(format!(
                    "{} where time <= to_timestamp({})",
                    sql, upper_bound
                ))
            }

            Expr::DurationLiteral(_) => Err(Error::InvalidFieldAccess(
                "cannot convert duration literal to SQL".to_string(),
            )),

            expr => {
                let field = expr.to_field()?;
                let table = expr.to_table()?;
                Ok(format!("select {} from {}", field, table))
            }
        }
    }
}

pub struct Entity {
    pub name: String,
    pub componnets: HashMap<String, Arc<EntityComponent>>,
}

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
                        .componnets
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
                    .into_iter()
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
                .into_iter()
                .map(|ast_node| self.parse(ast_node))
                .collect::<Result<Vec<_>, _>>()
                .map(Expr::Tuple),
            AstNode::StringLiteral(s) => {
                // Parse duration strings like "5m", "10s", "1h"
                self.parse_duration(s.as_ref()).map(Expr::DurationLiteral)
            }
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
                let mut suggestions = entity.componnets.keys().cloned().collect::<Vec<_>>();
                suggestions.sort();
                suggestions.push("time".to_string());
                suggestions
            }
            Expr::Component(component) => component
                .element_names
                .iter()
                .cloned()
                .chain(["last".to_string(), "first".to_string(), "time".to_string()])
                .collect(),
            Expr::Time(_) => {
                vec!["fftfreq".to_string()]
            }
            Expr::ArrayAccess(_, _) => {
                vec!["fft".to_string(), "last".to_string(), "first".to_string()]
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
            Expr::Last(_, _) => {
                vec![]
            }
            Expr::First(_, _) => {
                vec![]
            }
        }
    }

    pub fn get_string_suggestions(&self, input: &str) -> Vec<(String, String)> {
        if input.is_empty() {
            let mut entity_names = self.entities.keys().cloned().collect::<Vec<_>>();
            entity_names.sort();
            return entity_names
                .into_iter()
                .map(|n| (n.clone(), n.clone()))
                .collect();
        }

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
        let (start, input) = if valid_ast {
            ("", input)
        } else if input.contains(',') {
            let comma_pos = input.rfind(',').unwrap();
            input.split_at(comma_pos + 1)
        } else if input.starts_with('(') {
            let paren_pos = input.rfind('(').unwrap();
            input.split_at(paren_pos + 1)
        } else {
            ("", input)
        };

        if input.ends_with('.') {
            let query = &input[..input.len() - 1];

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
                vec![
                    AstNode::Ident(Cow::Borrowed("b")),
                    AstNode::Ident(Cow::Borrowed("c")),
                ]
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
            componnets: components,
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
        assert_eq!(result.unwrap(), "select world_pos from a_world_pos");
    }

    #[test]
    fn test_time_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let expr = Expr::Time(entity_component);
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select time from a_world_pos");
    }

    #[test]
    fn test_fftfreq_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        let time_expr = Expr::Time(entity_component);
        let expr = Expr::FftFreq(Box::new(time_expr));
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select fftfreq(time) from a_world_pos");
    }

    #[test]
    fn test_array_access_sql() {
        let entity_component = create_test_entity_component();
        let context = create_test_context();

        // Test first element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0);
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select world_pos[1] from a_world_pos");

        // Test second element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 1);
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select world_pos[2] from a_world_pos");

        // Test third element
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 2);
        let result = expr.to_sql(&context);
        assert_eq!(result.unwrap(), "select world_pos[3] from a_world_pos");
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
            "select time, world_pos[1] from a_world_pos"
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
            "select world_pos[1], world_pos[2], world_pos[3] from a_world_pos"
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
            "select a_world_pos.world_pos, b_velocity.velocity from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with Time from one table and Component from another
        let expr = Expr::Tuple(vec![
            Expr::Time(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, b_velocity.velocity from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with ArrayAccess from different tables
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component2.clone())), 1),
        ]);
        let result = expr.to_sql(&context);
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[1], b_velocity.velocity[2] from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
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
            "select a_world_pos.world_pos, b_velocity.velocity, c_acceleration.acceleration from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time JOIN c_acceleration ON a_world_pos.time = c_acceleration.time"
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
        let component = entity.componnets.get("component").unwrap();
        let expr = Expr::Component(component.clone());

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"first".to_string()));
        assert!(suggestions.contains(&"last".to_string()));
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
        assert_eq!(suggestions, vec!["fftfreq"]);
    }

    #[test]
    fn test_array_access_suggestions() {
        let component = create_test_entity_component();
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(component)), 0);
        let context = create_test_context();

        let suggestions = context.get_suggestions(&expr);
        assert!(suggestions.contains(&"fft".to_string()));
        assert!(suggestions.contains(&"first".to_string()));
        assert!(suggestions.contains(&"last".to_string()));
    }

    #[test]
    fn test_string_suggestions_empty() {
        let context = create_test_context();
        let suggestions = context.get_string_suggestions("");

        assert!(suggestions.contains(&"test_entity".to_string()));
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_string_suggestions_with_period() {
        let context = create_test_context();
        let suggestions = context.get_string_suggestions("test_entity.");

        assert!(suggestions.contains(&"component".to_string()));
        assert!(suggestions.contains(&"time".to_string()));
    }

    #[test]
    fn test_string_suggestions_component_with_period() {
        let context = create_test_context();
        let suggestions = context.get_string_suggestions("test_entity.component.");

        assert!(suggestions.contains(&"first".to_string()));
        assert!(suggestions.contains(&"last".to_string()));
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
