use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};
use syn::{Generics, Ident};

pub enum IdAttr {
    Id(String),
    Prefix(String),
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
            panic!("must include either prefix or id");
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
    let id_string = match id {
        IdAttr::Id(id) => {
            format!("elodin_conduit::cid!({})", id)
        }
        IdAttr::Prefix(prefix) => {
            let ident = ident.to_string().to_case(Case::Snake);
            format!("elodin_conduit::cid!({};{})", prefix, ident)
        }
    };
    let id: proc_macro2::TokenStream = id_string.parse().unwrap();
    if postcard {
        quote! {
            impl elodin_conduit::Component for #ident #generics #where_clause {
                fn component_id() -> elodin_conduit::ComponentId {
                    #id
                }

                fn component_type() -> elodin_conduit::ComponentType {
                    elodin_conduit::ComponentType::Bytes
                }

                fn component_value<'a>(&self) -> elodin_conduit::ComponentValue<'a> {
                    elodin_conduit::ComponentValue::Bytes(postcard::to_allocvec(self).unwrap_or_default().into())
                }

                fn from_component_value(value: elodin_conduit::ComponentValue<'_>) -> Option<Self> {
                    let elodin_conduit::ComponentValue::Bytes(buf) = value else {
                        return None;
                    };
                    postcard::from_bytes(buf.as_ref()).ok()
                }

            }
        }
        .into()
    } else {
        quote! {
            impl elodin_conduit::Component for #ident #generics #where_clause {
                fn component_id() -> elodin_conduit::ComponentId {
                    #id
                }

                fn component_type() -> elodin_conduit::ComponentType {
                    use elodin_conduit::Component;
                    <#ty>::component_type()
                }

                fn component_value<'a>(&self) -> elodin_conduit::ComponentValue<'a> {
                    use elodin_conduit::Component;
                    self.0.component_value()
                }

                fn from_component_value(value: elodin_conduit::ComponentValue<'_>) -> Option<Self> {
                    use elodin_conduit::Component;
                    <#ty>::from_component_value(value).map(Self)
                }

            }
        }
        .into()
    }
}
