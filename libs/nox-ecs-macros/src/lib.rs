use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::Ident;

extern crate proc_macro;

mod archetype;
mod component;
mod component_group;
mod from_builder;
mod into_op;

#[proc_macro_derive(Component, attributes(nox))]
pub fn component(input: TokenStream) -> TokenStream {
    component::component(input)
}

#[proc_macro_derive(Archetype, attributes(nox))]
pub fn archetype(input: TokenStream) -> TokenStream {
    archetype::archetype(input)
}

#[proc_macro_derive(FromBuilder, attributes(nox))]
pub fn from_builder(input: TokenStream) -> TokenStream {
    from_builder::from_builder(input)
}

#[proc_macro_derive(IntoOp, attributes(nox))]
pub fn into_op(input: TokenStream) -> TokenStream {
    into_op::into_op(input)
}

#[proc_macro_derive(ComponentGroup, attributes(nox))]
pub fn component_group(input: TokenStream) -> TokenStream {
    component_group::component_group(input)
}

pub(crate) fn nox_ecs_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("nox-ecs").expect("nox-ecs is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}

pub(crate) fn nox_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("nox").expect("nox is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
