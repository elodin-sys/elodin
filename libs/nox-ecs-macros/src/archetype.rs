use darling::ast::{self};
use darling::{FromDeriveInput, FromField};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

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
            fn components() -> Vec<(#crate_name::impeller2::schema::Schema<Vec<u64>>, #crate_name::impeller2_wkt::ComponentMetadata)> {
                use #crate_name::impeller2::component::Component;
                use #crate_name::archetype::ComponentExt;
                vec![#( (<#tys>::schema(), <#tys>::metadata()), )*]
            }

            fn insert_into_world(self, world: &mut #crate_name::World) {
                #(
                   self.#idents.insert_into_world(world);
                )*
            }
        }
    }
    .into()
}
