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
    entity_id: Option<u64>,
}

pub fn metadatatize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Metadatatize {
        ident,
        generics,
        data,
        entity_id,
    } = Metadatatize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller2 };
    let impeller_wkt = quote! { #crate_name::impeller2_wkt };
    let fields = data.take_struct().unwrap();

    let metadata_items = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let entity_id = field.entity_id.or(entity_id);
        if entity_id.is_some() {
            let name = field
                .ident
                .as_ref()
                .expect("only named field allowed")
                .to_string();
            let ident = &field.ident.as_ref().expect("field must have ident");
            let component_id = match &field.component_id {
                Some(c) => quote! {
                    #impeller::types::ComponentId::new(#c)
                },
                None => {
                    quote! {
                        #impeller::types::ComponentId::new(stringify!(#ident))
                    }
                }
            };
            let asset = field.asset.unwrap_or_default();
            quote! {
                .chain(core::iter::once(#impeller_wkt::ComponentMetadata {
                    component_id: #component_id,
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
