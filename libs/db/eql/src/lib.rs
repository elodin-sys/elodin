use std::{
    borrow::Cow,
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use impeller2::{
    schema::Schema,
    types::{ComponentId, EntityId},
};
use peg::error::ParseError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstNode<'input> {
    Ident(Cow<'input, str>),
    Field(Box<AstNode<'input>>, Cow<'input, str>),
    MethodCall(Box<AstNode<'input>>, Cow<'input, str>, Vec<AstNode<'input>>),
    Tuple(Vec<AstNode<'input>>),
}

peg::parser! {
    grammar ast_parser() for str {
        rule _ = quiet!{[' ' | '\n' | '\t']*}
        rule ident_str() -> Cow<'input, str> = s:$(['a'..='z' | 'A'..='Z' | '0'..='9' | '_']+) { Cow::Borrowed(s) }
        rule comma() = ("," _?)

        pub rule expr() -> AstNode<'input> = precedence! {
        e:(@) "." i:ident_str() "(" args:expr() ** comma() ")" { AstNode::MethodCall(Box::new(e), i,  args) }
        --
        "(" e:expr() ** comma() ")" { AstNode::Tuple(e) }
        --
        e:(@) "." i:ident_str() { AstNode::Field(Box::new(e), i) }
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
    pub fn to_sql(&self) -> Result<String, Error> {
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

            Expr::Last(_, _) => Err(Error::InvalidMethodCall(
                "last not supported in SQL conversion".to_string(),
            )),
            Expr::First(_, _) => Err(Error::InvalidMethodCall(
                "first not supported in SQL conversion".to_string(),
            )),

            // For single expressions, use the helper functions
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
}

impl Context {
    pub fn new(entities: HashMap<String, Arc<Entity>>) -> Self {
        Self { entities }
    }

    pub fn sql(&self, query: &str) -> Result<String, Error> {
        let ast = ast_parser::expr(query)?;
        let expr = self.parse(&ast)?;
        expr.to_sql()
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
                    _ => Err(Error::InvalidMethodCall(cow.to_string())),
                }
            }
            AstNode::Tuple(ast_nodes) => ast_nodes
                .into_iter()
                .map(|ast_node| self.parse(ast_node))
                .collect::<Result<Vec<_>, _>>()
                .map(Expr::Tuple),
        }
    }
}

// fn parse_swizzle(s: &str, shape: &[u64]) -> Result<usize, Error> {
//     let mut offset = 0;
//     for ((c, d), r) in s.chars().zip(shape.iter().cloned()).zip(
//         shape
//             .iter()
//             .take(shape.len().saturating_sub(1))
//             .cloned()
//             .chain(std::iter::once(1)),
//     ) {
//         let i = match c.to_ascii_lowercase() {
//             'x' => 0,
//             'y' => 1,
//             'z' => 2,
//             'w' => 3,
//             _ => return Err(Error::InvalidSwizzle(c)),
//         };
//         if i >= d {
//             return Err(Error::InvalidSwizzle(c));
//         }
//         offset += i * r;
//     }
//     Ok(offset as usize)
// }

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

    #[test]
    fn test_sql_from_expr() {
        use impeller2::types::{ComponentId, EntityId, PrimType};

        // Set up test data
        let entity_id = EntityId(1);
        let component_id = ComponentId::new("world_pos");

        // Create a test EntityComponent with name fields
        let entity_component = Arc::new(EntityComponent::new(
            "world_pos".to_string(),
            "a".to_string(),
            entity_id,
            component_id,
            Schema::new(PrimType::F64, vec![3u64]).unwrap(), // 3D vector schema
        ));

        // Test Component -> select component from entity_component
        let expr = Expr::Component(entity_component.clone());
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select world_pos from a_world_pos");

        // Test Time -> select time from entity_component
        let expr = Expr::Time(entity_component.clone());
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select time from a_world_pos");

        // Test fftfreq
        let expr = Expr::FftFreq(Box::new(expr));
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select fftfreq(time) from a_world_pos");

        // Test ArrayAccess -> select component[index] from entity_component
        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0);
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select world_pos[1] from a_world_pos");

        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 1);
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select world_pos[2] from a_world_pos");

        let expr = Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 2);
        let result = expr.to_sql();
        assert_eq!(result.unwrap(), "select world_pos[3] from a_world_pos");

        // Test Tuple with Time and ArrayAccess -> select time, component[index] from entity_component
        let expr = Expr::Tuple(vec![
            Expr::Time(entity_component.clone()),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
        ]);
        let result = expr.to_sql();
        assert_eq!(
            result.unwrap(),
            "select time, world_pos[1] from a_world_pos"
        );

        // Test Tuple with multiple ArrayAccess -> select component[1], component[2], component[3] from entity_component
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 1),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 2),
        ]);
        let result = expr.to_sql();
        assert_eq!(
            result.unwrap(),
            "select world_pos[1], world_pos[2], world_pos[3] from a_world_pos"
        );

        // Test JOIN functionality with multiple tables
        let entity_component2 = Arc::new(EntityComponent::new(
            "velocity".to_string(),
            "b".to_string(),
            EntityId(2),
            ComponentId::new("velocity"),
            Schema::new(PrimType::F64, vec![3u64]).unwrap(),
        ));

        // Test Tuple with elements from different tables -> JOIN
        let expr = Expr::Tuple(vec![
            Expr::Component(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
        ]);
        let result = expr.to_sql();
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos, b_velocity.velocity from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with Time from one table and Component from another -> JOIN
        let expr = Expr::Tuple(vec![
            Expr::Time(entity_component.clone()),
            Expr::Component(entity_component2.clone()),
        ]);
        let result = expr.to_sql();
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.time, b_velocity.velocity from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with ArrayAccess from different tables -> JOIN
        let expr = Expr::Tuple(vec![
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component.clone())), 0),
            Expr::ArrayAccess(Box::new(Expr::Component(entity_component2.clone())), 1),
        ]);
        let result = expr.to_sql();
        assert_eq!(
            result.unwrap(),
            "select a_world_pos.world_pos[1], b_velocity.velocity[2] from a_world_pos JOIN b_velocity ON a_world_pos.time = b_velocity.time"
        );

        // Test Tuple with three different tables
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
        let result = expr.to_sql();
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
}
