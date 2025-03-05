use convert_case::{Case, Casing};
use darling::FromDeriveInput;
use darling::ast::{self};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

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
    let name = name.unwrap_or(ident.to_string().to_case(Case::Snake));

    let comp_where = if let Some(where_clause) = generics.where_clause.clone() {
        quote! { #where_clause, Self: #crate_name::nox::ReprMonad<#crate_name::nox::Op>
        Self: #crate_name::nox::FromBuilder }
    } else {
        quote! { where Self: #crate_name::nox::ReprMonad<#crate_name::nox::Op> + for<'a> #crate_name::nox::FromBuilder<Item<'a> = Self> }
    };
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    quote! {
        impl #impl_generics #crate_name::impeller2::component::Component for #ident #ty_generics #where_clause {
            const NAME: &'static str = #name;

            fn schema() -> #crate_name::impeller2::schema::Schema<Vec<u64>> {
                <#ty as #crate_name::impeller2::component::Component>::schema()
            }
        }

        impl #impl_generics #crate_name::Component for #ident #ty_generics #comp_where
        {}
    }
    .into()
}
