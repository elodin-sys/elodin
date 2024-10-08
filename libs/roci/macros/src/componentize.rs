use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct Componentize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    entity_id: Option<u64>,
}

pub fn componentize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Componentize {
        ident,
        generics,
        data,
        entity_id,
    } = Componentize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller };
    let fields = data.take_struct().unwrap();
    let sink_calls = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let component_id = match &field.component_id {
            Some(c) => quote! {
                #crate_name::impeller::ComponentId::new(#c)
            },
            None => {
                quote! {
                    #crate_name::impeller::ComponentId::new(<#ty as #crate_name::impeller::Component>::NAME)
                }
            },
        };
        let ident = field.ident.as_ref().expect("only named fields allowed");
        if let Some(id) = field.entity_id.or(entity_id) {
            quote! {
                output.apply_value(
                    #component_id,
                     #impeller::EntityId(#id),
                    self.#ident.fixed_dim_component_value().clone(),
                );
            }
        }else{
            quote! {
                self.#ident.sink_columns(output);
            }
        }

    });
    let count_arms = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let entity_id = field.entity_id.or(entity_id);
        if entity_id.is_some() {
            quote! {
                <#ty as #crate_name::impeller::ConstComponent>::MAX_SIZE +
            }
        } else {
            quote! {
                <#ty as #crate_name::Componentize>::MAX_SIZE +
            }
        }
    });

    quote! {
        impl #crate_name::Componentize for #ident #generics #where_clause {
            fn sink_columns(&self, output: &mut impl #crate_name::Decomponentize) {
                use #impeller::ValueRepr;
                #(#sink_calls)*
            }

            const MAX_SIZE: usize = #(#count_arms)* 0;
        }
    }
    .into()
}
