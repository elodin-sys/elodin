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

pub fn into_op(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let FromBuilder {
        ident,
        generics,
        data,
    } = FromBuilder::from_derive_input(&input).unwrap();
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
    let fields = if fields.len() == 1 {
        let field_ident = &fields.fields[0].ident;
        quote! {
            use #crate_name::IntoOp;
            self.#field_ident.into_op()

        }
    } else {
        let fields = fields.fields.iter().map(|f| {
            let i = &f.ident;
            quote! {
                self.#i.into_op(),
            }
        });
        quote! {
            #crate_name::Noxpr::tuple(vec![#( #fields )*])
        }
    };
    let generic_idents = generics.type_params().map(|p| &p.ident);
    quote! {
        impl #generics #crate_name::IntoOp for #ident<#(#generic_idents,)*> #where_clause {
            fn into_op(self) -> #crate_name::Noxpr {
                #fields
            }
        }

    }
    .into()
}
