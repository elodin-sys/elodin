use proc_macro::TokenStream;

extern crate proc_macro;

mod component;
mod editable;

#[proc_macro_derive(Editable, attributes(editable))]
pub fn derive_macro_editable(input: TokenStream) -> TokenStream {
    editable::derive_proc_macro_impl(input)
}

#[proc_macro_derive(Component, attributes(conduit))]
pub fn component(input: TokenStream) -> TokenStream {
    component::component(input)
}

pub(crate) fn conduit_crate_name() -> proc_macro2::TokenStream {
    use proc_macro2::Span;
    use proc_macro_crate::{crate_name, FoundCrate};
    use quote::quote;
    use syn::Ident;

    let name = crate_name("elodin-conduit").expect("my-crate is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
