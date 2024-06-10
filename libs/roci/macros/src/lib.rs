use darling::FromField;
use proc_macro::TokenStream;
use proc_macro2::Span;
use proc_macro_crate::{crate_name, FoundCrate};
use quote::quote;
use syn::Ident;

mod componentize;
mod decomponentize;

#[derive(Debug, FromField)]
#[darling(attributes(roci))]
struct Field {
    ident: Option<syn::Ident>,
    ty: syn::Type,
    entity_id: Option<u64>,
    component_id: Option<String>,
}

#[proc_macro_derive(Componentize, attributes(roci))]
pub fn componentize(input: TokenStream) -> TokenStream {
    componentize::componentize(input)
}

#[proc_macro_derive(Decomponentize, attributes(roci))]
pub fn decomponentize(input: TokenStream) -> TokenStream {
    decomponentize::decomponentize(input)
}

pub(crate) fn roci_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("roci").expect("roci is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
