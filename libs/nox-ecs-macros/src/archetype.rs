use darling::ast::{self};
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_named))]
pub struct Archetype {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), Field>,
}

#[derive(Debug, FromField)]
struct Field {
    ident: Option<syn::Ident>,
    ty: syn::Type,
}

pub fn archetype(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_ecs_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Archetype {
        ident,
        generics,
        data,
    } = Archetype::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
    let tys = fields.iter().map(|f| f.ty.clone()).collect::<Vec<_>>();
    let idents = fields
        .iter()
        .map(|f| f.ident.clone().unwrap())
        .collect::<Vec<_>>();
    let where_clause = &generics.where_clause;
    quote! {
        impl #crate_name::Archetype for #ident #generics #where_clause {
            fn component_ids() -> Vec<#crate_name::elodin_conduit::ComponentId> {
                use #crate_name::Component;
                vec![#( <#tys>::component_id(), )*]
            }

            fn component_tys() -> Vec<#crate_name::elodin_conduit::ComponentType> {
                use #crate_name::Component;
                vec![#( <#tys>::component_type(), )*]
            }

            fn insert_into_table(self, table: &mut #crate_name::Table<#crate_name::HostStore>) {
                #(
                   self.#idents.insert_into_table(table);
                )*
            }
        }
    }
    .into()
}
