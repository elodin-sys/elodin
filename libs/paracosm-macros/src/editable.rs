use std::ops::RangeInclusive;

use darling::ast::NestedMeta;
use darling::Error;
use darling::FromMeta;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::Expr;
use syn::Lit;
use syn::RangeLimits;
use syn::UnOp;
use syn::{parse_macro_input, punctuated::Punctuated, DeriveInput, Meta, Token};

#[derive(Debug)]
struct InputRange(RangeInclusive<f64>);

impl InputRange {
    fn parse_boundary_lit(lit: &syn::Lit, neg: bool) -> darling::Result<f64> {
        let multi = if neg { -1_f64 } else { 1_f64 };
        match lit {
            Lit::Float(ref lit) => Ok(multi * lit.base10_parse::<f64>().unwrap()),
            Lit::Int(ref lit) => Ok(multi * lit.base10_parse::<f64>().unwrap()),
            _ => Err(Error::unexpected_lit_type(lit)),
        }
    }

    fn get_boundary(expr: &syn::Expr) -> darling::Result<f64> {
        match *expr {
            Expr::Unary(ref u) => match u.op {
                UnOp::Neg(_) => match *u.expr {
                    Expr::Lit(ref l) => Self::parse_boundary_lit(&l.lit, true),
                    _ => Err(Error::unexpected_expr_type(expr)),
                },
                _ => Err(Error::unexpected_expr_type(expr)),
            },
            Expr::Lit(ref l) => Self::parse_boundary_lit(&l.lit, false),
            _ => Err(Error::unexpected_expr_type(expr)),
        }
    }
}

impl FromMeta for InputRange {
    fn from_expr(expr: &syn::Expr) -> darling::Result<Self> {
        match *expr {
            Expr::Range(ref range) => match range.limits {
                RangeLimits::Closed(_) => {
                    let start = Self::get_boundary(range.start.as_ref().unwrap())?;
                    let end = Self::get_boundary(range.end.as_ref().unwrap())?;
                    Ok(InputRange(start..=end))
                }
                _ => Err(Error::unexpected_type("Only support RangeInclusive<f64>")),
            },
            _ => Err(Error::unexpected_expr_type(expr)),
        }
    }
}

#[derive(Debug, FromMeta)]
pub struct EditableAttributes {
    #[darling(default)]
    slider: bool,
    range: InputRange,
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
            range,
            name,
            ..
        } = editable_attributes;

        let range_min = range.0.start();
        let range_max = range.0.end();

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
                fn build(&mut self, ui: &mut Ui) {
                    let mut num = self.0.load();
                    ui.add(egui::Slider::new(num.deref_mut(), #range_min..=#range_max).text(#name));
                }
            }
        }
        .into()
    } else {
        quote! {
            impl Editable for #item_identifier #generics #where_clause {
                fn build(&mut self, ui: &mut Ui) {
                    let mut num = self.0.load();
                    ui.add(egui::Slider::new(num.deref_mut(), -1.25..=1.25).text("input"));
                }
            }
        }
        .into()
    }
}
