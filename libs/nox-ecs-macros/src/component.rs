use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct Component {
    id: Option<String>,
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Type>,
}

pub fn component(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_ecs_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Component {
        id,
        ident,
        generics,
        data,
    } = Component::from_derive_input(&input).unwrap();
    let id_string = id.unwrap_or_else(|| ident.to_string().to_case(Case::Snake));
    let id = quote! { #crate_name::elodin_conduit::ComponentId::new(#id_string) };
    let fields = data.take_struct().unwrap();
    let ty = &fields.fields[0];
    let where_clause = &generics.where_clause;
    quote! {
        impl #crate_name::nox::IntoOp for #ident #generics #where_clause {
            fn into_op(self) -> #crate_name::nox::Noxpr {
                use #crate_name::nox::IntoOp;
                self.0.into_op()
            }
        }

        impl #crate_name::nox::FromBuilder for #ident #generics #where_clause {
            type Item<'a> = Self;

            fn from_builder(builder: &#crate_name::nox::Builder) -> Self::Item<'_> {
                Self(<#ty as #crate_name::nox::FromBuilder>::from_builder(builder))
            }
        }


        impl #crate_name::Component for #ident #generics #where_clause {
            type Inner = #ty;
            type HostTy = <#ty as #crate_name::Component>::HostTy;

            fn host(val: Self::HostTy) -> Self {
                Self(<#ty as #crate_name::Component>::host(val))
            }

            fn component_id() -> #crate_name::elodin_conduit::ComponentId {
                #id
            }

            fn component_type() -> #crate_name::elodin_conduit::ComponentType {
                <#ty as #crate_name::Component>::component_type()
            }
        }
    }
    .into()
}
