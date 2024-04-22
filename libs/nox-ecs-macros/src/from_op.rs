use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct FromOp {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Field>,
}

pub fn from_op(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let FromOp {
        ident,
        generics,
        data,
    } = FromOp::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
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
    if fields.len() != 1 {
        panic!("FromOp only supports structs with a single field");
    }
    let field_ident = &fields.fields[0].ident;
    let field_ty = &fields.fields[0].ty;
    let generic_idents = generics.type_params().map(|p| &p.ident);
    quote! {
        impl #generics #crate_name::FromOp for #ident<#(#generic_idents,)*> #where_clause {
            fn from_op(noxpr: #crate_name::Noxpr) -> Self {
                use #crate_name::FromOp;
                Self {
                    #field_ident: <#field_ty as #crate_name::FromOp>::from_op(noxpr)
                }
            }
        }
    }
    .into()
}
