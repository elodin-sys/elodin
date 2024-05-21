use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct Component {
    name: Option<String>,
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Type>,
}

pub fn component(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_ecs_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Component {
        name,
        ident,
        generics,
        data,
    } = Component::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
    let ty = &fields.fields[0];
    let where_clause = &generics.where_clause;
    let name = name.unwrap_or(ident.to_string().to_case(Case::Snake));
    quote! {
        impl #crate_name::nox::IntoOp for #ident #generics #where_clause {
            fn into_op(self) -> #crate_name::nox::Noxpr {
                use #crate_name::nox::IntoOp;
                self.0.into_op()
            }
        }

        impl #crate_name::nox::FromOp for #ident #generics #where_clause {
            fn from_op(noxpr: #crate_name::nox::Noxpr) -> Self {
                Self(<#ty as #crate_name::nox::FromOp>::from_op(noxpr))
            }
        }

        impl #crate_name::nox::FromBuilder for #ident #generics #where_clause {
            type Item<'a> = Self;

            fn from_builder(builder: &#crate_name::nox::Builder) -> Self::Item<'_> {
                Self(<#ty as #crate_name::nox::FromBuilder>::from_builder(builder))
            }
        }


        impl #crate_name::conduit::Component for #ident #generics #where_clause {
            fn name() -> String {
                #name.to_string()
            }

            fn component_type() -> #crate_name::conduit::ComponentType {
                <#ty as #crate_name::conduit::Component>::component_type()
            }
        }

        impl #crate_name::Component for #ident #generics #where_clause {}
    }
    .into()
}
