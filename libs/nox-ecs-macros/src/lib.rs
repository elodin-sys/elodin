use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::quote;
use syn::{Ident, TypeParamBound};

extern crate proc_macro;

mod archetype;
mod component;
mod component_group;
mod from_builder;
mod from_op;
mod into_op;
mod repr_monad;

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

#[proc_macro_derive(FromOp, attributes(nox))]
pub fn from_op(input: TokenStream) -> TokenStream {
    from_op::from_op(input)
}

#[proc_macro_derive(ComponentGroup, attributes(nox))]
pub fn component_group(input: TokenStream) -> TokenStream {
    component_group::component_group(input)
}

pub(crate) fn nox_ecs_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("nox-ecs");

    match name {
        Ok(FoundCrate::Name(name)) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
        _ => quote!(::nox_ecs),
    }
}

#[proc_macro_derive(ReprMonad, attributes(nox))]
pub fn repr_monad(input: TokenStream) -> TokenStream {
    repr_monad::repr_monad(input)
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

pub(crate) fn is_repr_bound(bound: &TypeParamBound) -> bool {
    let TypeParamBound::Trait(t) = bound else {
        return false;
    };
    t.path.is_ident("Repr")
        || t.path.is_ident("nox::Repr")
        || t.path.is_ident("nox::OwnedRepr")
        || t.path.is_ident("OwnedRepr")
}
