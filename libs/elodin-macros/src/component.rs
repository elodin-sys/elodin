use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};
use syn::{Generics, Ident};

use crate::conduit_crate_name;

pub enum IdAttr {
    Id(String),
    Prefix(String),
    Ident,
}

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(conduit), supports(struct_tuple, struct_named))]
pub struct Component {
    id: Option<String>,
    prefix: Option<String>,
    #[darling(default)]
    postcard: bool,
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), FieldRecv>,
}

impl Component {
    fn id(&self) -> IdAttr {
        if let Some(id) = &self.id {
            IdAttr::Id(id.clone())
        } else if let Some(prefix) = &self.prefix {
            IdAttr::Prefix(prefix.clone())
        } else {
            IdAttr::Ident
        }
    }
}

#[derive(Debug, FromField)]
struct FieldRecv {
    ty: syn::Type,
}

pub fn component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let component = Component::from_derive_input(&input).unwrap();
    let id = component.id();
    let Component {
        ident,
        generics,
        data,
        postcard,
        ..
    } = component;
    let fields = data.take_struct().unwrap();
    let ty = &fields.fields[0].ty;
    let where_clause = &generics.where_clause;
    let crate_name = conduit_crate_name();
    let id_string = match id {
        IdAttr::Id(id) => {
            format!("{crate_name}::cid!({id})",)
        }
        IdAttr::Prefix(prefix) => {
            let ident = ident.to_string().to_case(Case::Snake);
            format!("{crate_name}::cid!({prefix};{ident})")
        }
        IdAttr::Ident => {
            let ident = ident.to_string().to_case(Case::Snake);
            format!("{crate_name}::ComponentId::new(\"{ident}\")")
        }
    };
    let id: proc_macro2::TokenStream = id_string.parse().unwrap();
    if postcard {
        quote! {
            impl #crate_name::Component for #ident #generics #where_clause {
                fn component_id() -> #crate_name::ComponentId {
                    #id
                }

                fn component_type() -> #crate_name::ComponentType {
                    #crate_name::ComponentType::Bytes
                }

                fn component_value<'a>(&self) -> #crate_name::ComponentValue<'a> {
                    #crate_name::ComponentValue::Bytes(postcard::to_allocvec(self).unwrap_or_default().into())
                }

                fn from_component_value(value: #crate_name::ComponentValue<'_>) -> Option<Self> {
                    let #crate_name::ComponentValue::Bytes(buf) = value else {
                        return None;
                    };
                    postcard::from_bytes(buf.as_ref()).ok()
                }

            }
        }
        .into()
    } else {
        quote! {
            impl #crate_name::Component for #ident #generics #where_clause {
                fn component_id() -> #crate_name::ComponentId {
                    #id
                }

                fn component_type() -> #crate_name::ComponentType {
                    use #crate_name::Component;
                    <#ty>::component_type()
                }

                fn component_value<'a>(&self) -> #crate_name::ComponentValue<'a> {
                    use #crate_name::Component;
                    self.0.component_value()
                }

                fn from_component_value(value: #crate_name::ComponentValue<'_>) -> Option<Self> {
                    use #crate_name::Component;
                    <#ty>::from_component_value(value).map(Self)
                }

            }
        }
        .into()
    }
}
