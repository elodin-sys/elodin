use darling::ast::NestedMeta;
use darling::Error;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, punctuated::Punctuated, DeriveInput, Meta, Token};

#[derive(Debug, FromMeta)]
pub struct EditableAttributes {
    #[darling(default)]
    slider: bool,
    #[darling(default)]
    range_min: f64,
    #[darling(default)]
    range_max: f64,
    name: String,
}

pub fn derive_proc_macro_impl(input: TokenStream) -> TokenStream {
    let DeriveInput {
        ident: item_identifier,
        data: _,
        generics,
        attrs,
        ..
    } = parse_macro_input!(input as DeriveInput);

    let where_clause = &generics.where_clause;

    //--------------------------------------------------------------------------

    let _attr = attrs.iter().find(|attr| attr.path().is_ident("editable"));

    if let Some(attr) = _attr {
        let nested = attr
            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .unwrap();

        let attr_args = match NestedMeta::parse_meta_list(nested.to_token_stream()) {
            Ok(v) => v,
            Err(e) => {
                return TokenStream::from(Error::from(e).write_errors());
            }
        };

        let editable_attributes = match EditableAttributes::from_list(&attr_args) {
            Ok(v) => v,
            Err(e) => {
                return TokenStream::from(e.write_errors());
            }
        };

        let EditableAttributes {
            slider,
            range_min,
            range_max,
            name,
            ..
        } = editable_attributes;

        if !slider {
            return syn::Error::new(
                item_identifier.span(),
                "Unknown `editable` element type (currently only `slider` is supported).",
            )
            .into_compile_error()
            .into();
        }

        quote! {
            impl Editable for #item_identifier #generics #where_clause {
                fn build(&mut self, ui: &mut bevy_egui::egui::Ui) {
                    use std::ops::DerefMut;
                    let mut num = self.0.load();
                    ui.add(bevy_egui::egui::Slider::new(num.deref_mut(), #range_min..=#range_max).text(#name));
                }
            }
        }
        .into()
    } else {
        quote! {
            impl Editable for #item_identifier #generics #where_clause {
                fn build(&mut self, ui: &mut bevy_egui::egui::Ui) {
                    use std::ops::DerefMut;
                    let mut num = self.0.load();
                    ui.add(bevy_egui::egui::Slider::new(num.deref_mut(), -1.25..=1.25).text("input"));
                }
            }
        }
        .into()
    }
}
