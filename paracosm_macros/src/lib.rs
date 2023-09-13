use proc_macro::TokenStream;

extern crate proc_macro;

mod editable;

#[proc_macro_derive(Editable, attributes(editable))]
pub fn derive_macro_editable(input: TokenStream) -> TokenStream {
    editable::derive_proc_macro_impl(input)
}
