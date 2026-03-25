use darling::FromField;
use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::quote;
use syn::Ident;

mod as_vtable;
mod componentize;
mod decomponentize;
mod metadatatize;

#[derive(Debug, FromField)]
#[darling(attributes(db))]
struct Field {
    ident: Option<syn::Ident>,
    ty: syn::Type,
    component_id: Option<String>,
    #[darling(skip, default)]
    nest: bool,
    #[darling(default)]
    timestamp: bool,
    #[darling(default)]
    skip: bool,
}

impl Field {
    pub fn component_id(&self) -> proc_macro2::TokenStream {
        let impeller = crate::impeller_crate_name();
        match &self.component_id {
            Some(c) => quote! {
                #impeller::types::ComponentId::new(#c)
            },
            None => {
                let ident = self.ident.as_ref().expect("field must have ident");
                quote! {
                    #impeller::types::ComponentId::new(stringify!(#ident))
                }
            }
        }
    }

    pub fn component_id_str(&self) -> String {
        match &self.component_id {
            Some(c) => c.clone(),
            None => self
                .ident
                .as_ref()
                .expect("field must have ident")
                .to_string(),
        }
    }
}

#[proc_macro_derive(Componentize, attributes(db))]
pub fn componentize(input: TokenStream) -> TokenStream {
    componentize::componentize(input)
}

#[proc_macro_derive(Decomponentize, attributes(db))]
pub fn decomponentize(input: TokenStream) -> TokenStream {
    decomponentize::decomponentize(input)
}

#[proc_macro_derive(Metadatatize, attributes(db))]
pub fn metadatize(input: TokenStream) -> TokenStream {
    metadatatize::metadatatize(input)
}

#[proc_macro_derive(AsVTable, attributes(db))]
pub fn as_vtable(input: TokenStream) -> TokenStream {
    as_vtable::as_vtable(input)
}

pub(crate) fn impeller_crate_name() -> proc_macro2::TokenStream {
    let name =
        crate_name("impeller2").expect("impeller2 must be a dependency to use db-macros derives");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}

pub(crate) fn wkt_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("impeller2-wkt")
        .expect("impeller2-wkt must be a dependency to use db-macros derives");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
