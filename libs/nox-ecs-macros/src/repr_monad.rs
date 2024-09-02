use crate::is_repr_bound;
use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct ReprMonad {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Field>,
}

pub fn repr_monad(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let ReprMonad {
        ident,
        generics,
        data,
    } = ReprMonad::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
    let inner_type = fields
        .fields
        .iter()
        .map(|f| f.ty.clone())
        .next()
        .expect("ReprMonad requires exactly one inner field");
    let inner_field = fields
        .fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if let Some(field_ident) = &f.ident {
                quote! { #field_ident }
            } else {
                let i = syn::Index::from(i);
                quote! { #i }
            }
        })
        .next()
        .expect("ReprMonad requires exactly one inner field");
    let where_clause_predicates = &generics.where_clause.as_ref().map(|w| &w.predicates);
    let repr_generic = generics
        .params
        .iter()
        .filter_map(|p| {
            let syn::GenericParam::Type(t) = p else {
                return None;
            };
            Some(t)
        })
        .find(|t| t.bounds.iter().any(is_repr_bound))
        .expect("repr generic argument required to impl ReprMonad");
    let repr_generic_ident = repr_generic.ident.clone();
    let bounds = generics
        .params
        .iter()
        .filter_map(|p| {
            let syn::GenericParam::Type(t) = p else {
                return None;
            };
            if t.bounds.iter().any(is_repr_bound) {
                return None;
            }
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

    let replaced_generic_idents: Vec<_> = generics
        .params
        .iter()
        .map(|p| {
            let syn::GenericParam::Type(t) = p else {
                return quote! { #p, };
            };
            if t.bounds.iter().any(is_repr_bound) {
                quote! { NewRepr, }
            } else {
                let ident = &t.ident;
                quote! {#ident, }
            }
        })
        .collect();
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    quote! {
        impl #impl_generics #crate_name::ReprMonad<#repr_generic_ident> for #ident #ty_generics #where_clause {
            type Elem = <#inner_type as #crate_name::ReprMonad<#repr_generic_ident>>::Elem;
            type Dim = <#inner_type as #crate_name::ReprMonad<#repr_generic_ident>>::Dim;
            type Map<NewRepr: Repr> = #ident<#(#replaced_generic_idents)*>;

            fn map<N: Repr>(
                self,
                func: impl Fn(R::Inner<Self::Elem, Self::Dim>) -> N::Inner<Self::Elem, Self::Dim>,
            ) -> Self::Map<N> {
                use #crate_name::ReprMonad;
                todo!()
                //#ident::from_inner(func(Self::into_inner()))
            }

            fn into_inner(self) -> R::Inner<Self::Elem, Self::Dim> {
                use #crate_name::ReprMonad;
                self.#inner_field.into_inner()
            }
            fn inner(&self) -> &R::Inner<Self::Elem, Self::Dim> {
                use #crate_name::ReprMonad;
                self.#inner_field.inner()
            }

            fn from_inner(inner: R::Inner<Self::Elem, Self::Dim>) -> Self {
                use #crate_name::ReprMonad;
                Self {
                    #inner_field: <#inner_type>::from_inner(inner)
                }
            }
        }
    }
    .into()
}
