use darling::FromDeriveInput;
use darling::ast;
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct Metadatatize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    parent: Option<String>,
}

pub fn metadatatize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Metadatatize {
        ident,
        generics,
        data,
        parent,
    } = Metadatatize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller2 };
    let impeller_wkt = quote! { #crate_name::impeller2_wkt };
    let fields = data.take_struct().unwrap();

    let metadata_items = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        if !field.nest {
            let name = field
                .ident
                .as_ref()
                .expect("only named field allowed")
                .to_string();
            let component_id = field.component_id();

            let component_id = if let Some(parent) = &parent {
                format!("{parent}.{component_id}")
            } else {
                component_id.to_string()
            };
            let asset = field.asset.unwrap_or_default();
            quote! {
                .chain(core::iter::once(#impeller_wkt::ComponentMetadata {
                    component_id: #impeller::types::ComponentId::new(#component_id),
                    name: #name.to_string(),
                    metadata: Default::default(),
                    asset: #asset,
                }))
            }
        } else {
            quote! {
                .chain(<#ty as #crate_name::Metadatatize>::metadata())
            }
        }
    });
    quote! {
        impl #crate_name::Metadatatize for #ident #generics #where_clause {
            fn metadata() -> impl Iterator<Item = #impeller_wkt::ComponentMetadata> {
                core::iter::empty()
                #(#metadata_items)*
            }
        }
    }
    .into()
}
