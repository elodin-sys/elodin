use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct Decomponentize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    entity_id: Option<u64>,
}

pub fn decomponentize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Decomponentize {
        ident,
        generics,
        data,
        entity_id,
    } = Decomponentize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller2 };
    let fields = data.take_struct().unwrap();
    let if_arms = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let component_id = match &field.component_id {
            Some(c) => quote! {
                #impeller::types::ComponentId::new(#c)
            },
            None => {
                quote! {
                    <#ty as #impeller::component::Component>::COMPONENT_ID
                }
            }
        };
        let ident = &field.ident;
        let name = field
            .ident
            .as_ref()
            .expect("only named field allowed")
            .to_string()
            .to_case(Case::UpperSnake);
        if let Some(id) = field.entity_id.or(entity_id) {
            let const_name = format!("{name}_ID");
            let const_name = syn::Ident::new(&const_name, Span::call_site());
            quote! {
                const #const_name: #impeller::types::ComponentId = #component_id;
                if component_id == #const_name && entity_id == #impeller::types::EntityId(#id) {
                    if let Ok(val) = <#ty as #impeller::com_de::FromComponentView>::from_component_view(view.clone()) {
                        self.#ident = val;
                    }
                }
            }
        } else {
            quote! {
                self.#ident.apply_value(component_id, entity_id, value.clone());
            }
        }
    });
    quote! {
        impl #crate_name::Decomponentize for #ident #generics #where_clause {
            fn apply_value(&mut self,
                            component_id: #impeller::types::ComponentId,
                            entity_id: #impeller::types::EntityId,
                            view: #impeller::types::ComponentView<'_>
            ) {
                #(#if_arms)*
            }
        }
    }
    .into()
}
