use convert_case::{Case, Casing};
use darling::FromDeriveInput;
use darling::ast::{self};
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct Decomponentize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    parent: Option<String>,
}

pub fn decomponentize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Decomponentize {
        ident,
        generics,
        data,
        parent,
    } = Decomponentize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller2 };
    let fields = data.take_struct().unwrap();
    let if_arms = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let ident = &field.ident;
        let name = field
            .ident
            .as_ref()
            .expect("only named field allowed")
            .to_string()
            .to_case(Case::UpperSnake);
        let component_id = field.component_id();

        let component_id = if let Some(parent) = &parent {
            format!("{parent}.{component_id}")
        } else {
            component_id.to_string()
        };
        let component_id = quote! { #impeller::types::ComponentId::new(#component_id) };
        if !field.nest {
        let const_name = format!("{name}_ID");
            let const_name = syn::Ident::new(&const_name, Span::call_site());
            quote! {
                const #const_name: #impeller::types::ComponentId = #component_id;
                if component_id == #const_name {
                    if let Ok(val) = <#ty as #impeller::com_de::FromComponentView>::from_component_view(view.clone()) {
                        self.#ident = val;
                    }
                }
            }
        } else {
            quote! {
                self.#ident.apply_value(component_id,  value.clone(), timestamp.clone())?;
            }
        }
    });
    quote! {
        impl #crate_name::Decomponentize for #ident #generics #where_clause {
            type Error = core::convert::Infallible;
            fn apply_value(&mut self,
                            component_id: #impeller::types::ComponentId,
                            view: #impeller::types::ComponentView<'_>,
                            timestamp: Option<#impeller::types::Timestamp>
            ) -> Result<(), Self::Error>{
                #(#if_arms)*
                Ok(())
            }
        }
    }
    .into()
}
