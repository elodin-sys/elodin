use darling::FromDeriveInput;
use darling::ast::{self};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(db), supports(struct_named))]
pub struct Componentize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    parent: Option<String>,
}

pub fn componentize(input: TokenStream) -> TokenStream {
    let impeller = crate::impeller_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Componentize {
        ident,
        generics,
        data,
        parent,
    } = Componentize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let fields = data.take_struct().unwrap();
    let sink_calls = fields.fields.iter().map(|field| {
        let component_id = field.component_id();

        let component_id = if let Some(parent) = &parent {
            format!("{parent}.{component_id}")
        } else {
            component_id.to_string()
        };
        let ident = field.ident.as_ref().expect("only named fields allowed");
        if !field.nest {
            quote! {
                let _ = output.apply_value(
                    #impeller::types::ComponentId::new(#component_id),
                    self.#ident.as_component_view(),
                    None
                );
            }
        } else {
            quote! {
                self.#ident.sink_columns(output);
            }
        }
    });

    quote! {
        impl #impeller::com_de::Componentize for #ident #generics #where_clause {
            fn sink_columns(&self, output: &mut impl #impeller::com_de::Decomponentize) {
                use #impeller::com_de::AsComponentView;
                #(#sink_calls)*
            }

            const MAX_SIZE: usize = 0;
        }
    }
    .into()
}
