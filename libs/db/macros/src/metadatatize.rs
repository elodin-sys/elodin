use darling::FromDeriveInput;
use darling::ast;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(db), supports(struct_named))]
pub struct Metadatatize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    parent: Option<String>,
}

pub fn metadatatize(input: TokenStream) -> TokenStream {
    let impeller = crate::impeller_crate_name();
    let impeller_wkt = crate::wkt_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Metadatatize {
        ident,
        generics,
        data,
        parent,
    } = Metadatatize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let fields = data.take_struct().unwrap();

    let metadata_items = fields
        .fields
        .iter()
        .filter(|f| !f.timestamp && !f.skip)
        .map(|field| {
            let ty = &field.ty;
            if !field.nest {
                let name = field
                    .ident
                    .as_ref()
                    .expect("only named field allowed")
                    .to_string();
                let name = if let Some(parent) = &parent {
                    format!("{parent}.{name}")
                } else {
                    name
                };
                let component_id = field.component_id();

                let component_id = if let Some(parent) = &parent {
                    format!("{parent}.{component_id}")
                } else {
                    component_id.to_string()
                };
                quote! {
                    .chain(core::iter::once(#impeller_wkt::ComponentMetadata {
                        component_id: #impeller::types::ComponentId::new(#component_id),
                        name: #name.to_string(),
                        metadata: Default::default(),
                    }))
                }
            } else {
                quote! {
                    .chain(<#ty as #impeller_wkt::Metadatatize>::metadata())
                }
            }
        });
    quote! {
        impl #impeller_wkt::Metadatatize for #ident #generics #where_clause {
            fn metadata() -> impl Iterator<Item = #impeller_wkt::ComponentMetadata> {
                core::iter::empty()
                #(#metadata_items)*
            }
        }
    }
    .into()
}
