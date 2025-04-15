use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::quote;
use syn::Ident;
use syn::{ItemFn, parse_macro_input};

/// Attribute macro for stellarator main functions that wraps an async function with stellarator::run
#[proc_macro_attribute]
pub fn main(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);

    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "the #[stellarator::main] attribute can only be applied to async functions",
        )
        .to_compile_error()
        .into();
    }

    let fn_name = &input_fn.sig.ident;
    let fn_body = &input_fn.block;
    let fn_vis = &input_fn.vis;
    let fn_attrs = &input_fn.attrs;
    let fn_generics = &input_fn.sig.generics;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;
    let stellar = stellar_crate_name();

    let result = quote! {
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_generics(#fn_inputs) #fn_output {
            #stellar::run(move || async move #fn_body)
        }
    };

    result.into()
}

/// Attribute macro for stellarator tests that wraps an async function with stellarator::run
#[proc_macro_attribute]
pub fn test(_args: TokenStream, input: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(input as ItemFn);

    if input_fn.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "the #[stellarator::test] attribute can only be applied to async functions",
        )
        .to_compile_error()
        .into();
    }

    let fn_name = &input_fn.sig.ident;
    let fn_body = &input_fn.block;
    let fn_vis = &input_fn.vis;
    let fn_attrs = &input_fn.attrs;
    let fn_generics = &input_fn.sig.generics;
    let stellar = stellar_crate_name();

    let result = quote! {
        #[::core::prelude::v1::test]
        #(#fn_attrs)*
        #fn_vis fn #fn_name #fn_generics() {
            #stellar::run(move || async move #fn_body)
        }
    };

    result.into()
}

pub(crate) fn stellar_crate_name() -> proc_macro2::TokenStream {
    let name = crate_name("stellarator").expect("stellarator is present in `Cargo.toml`");

    match name {
        FoundCrate::Itself => quote!(crate),
        FoundCrate::Name(name) => {
            let ident = Ident::new(&name, Span::call_site());
            quote!( #ident )
        }
    }
}
