use proc_macro::TokenStream;

extern crate proc_macro;

mod archetype;
mod component;

#[proc_macro_derive(Component, attributes(nox))]
pub fn component(input: TokenStream) -> TokenStream {
    component::component(input)
}

#[proc_macro_derive(Archetype, attributes(nox))]
pub fn archetype(input: TokenStream) -> TokenStream {
    archetype::archetype(input)
}

pub(crate) fn nox_ecs_crate_name() -> proc_macro2::TokenStream {
    use proc_macro2::Span;
    use proc_macro_crate::{crate_name, FoundCrate};
    use quote::quote;
    use syn::Ident;

    let name = crate_name("nox-ecs").expect("my-crate is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
