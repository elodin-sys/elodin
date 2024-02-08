use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct FromBuilder {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Field>,
}

pub fn from_builder(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let FromBuilder {
        ident,
        generics,
        data,
    } = FromBuilder::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
    let fields = fields.fields.iter().map(|f| {
        let ty = &f.ty;
        let field_ident = &f.ident;
        quote! {
            #field_ident: <#ty as #crate_name::FromBuilder>::from_builder(builder),
        }
    });
    let where_clause_predicates = &generics.where_clause.as_ref().map(|w| &w.predicates);
    let bounds = generics
        .params
        .iter()
        .filter_map(|p| {
            let syn::GenericParam::Type(t) = p else {
                return None;
            };
            let ident = &t.ident;
            Some(quote! { #ident: #crate_name::xla::ArrayElement + #crate_name::xla::NativeType, })
        })
        .collect::<Vec<_>>();
    let where_clause = if where_clause_predicates.is_some() || !bounds.is_empty() {
        quote! {
            where #(#bounds)* #where_clause_predicates
        }
    } else {
        quote! {}
    };
    quote! {
          impl #generics #crate_name::FromBuilder for #ident #generics #where_clause {
            type Item<'a> = Self;

            fn from_builder(builder: &#crate_name::Builder) -> Self::Item<'_> {
                Self {
                    #(#fields)*
                }
            }
        }
    }
    .into()
}
