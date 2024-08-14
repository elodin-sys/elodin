use convert_case::{Case, Casing};
use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct Metadatatize {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
}

pub fn metadatatize(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let Metadatatize {
        ident,
        generics,
        data,
    } = Metadatatize::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let conduit = quote! { #crate_name::conduit };
    let fields = data.take_struct().unwrap();
    let ids = fields.fields.iter().map(|field| {
        let ident = field.ident.as_ref().expect("only named fields allowed");
        if field.entity_id.is_some() {
            let component_id = match &field.component_id {
                Some(c) => quote! {
                    #crate_name::conduit::ComponentId::new(#c)
                },
                None => {
                    let ty = &field.ty;
                    quote! {
                        #crate_name::conduit::ComponentId::new(<#ty as #crate_name::conduit::Component>::NAME)
                    }
                },
            };
            let const_name = field.ident.as_ref().expect("only named field allowed").to_string().to_case(Case::UpperSnake);
            let const_name = format!("{const_name}_ID");
            let const_name = syn::Ident::new(&const_name, Span::call_site());
            quote! {
                const #const_name: #conduit::ComponentId =  #component_id;
            }
        }else{
            quote! {
                if let Some(metadata) = self.#ident.get_metadata(component_id) {
                    return Some(metadata);
                }
            }
        }
    });
    let match_arms = fields.fields.iter().map(|field| {
        if field.entity_id.is_some() {
            let ty = &field.ty;
            let component_name = match &field.component_id {
                Some(c) => quote! {
                    #c
                },
                None => {
                    quote! {
                        <#ty as #crate_name::conduit::Component>::NAME
                    }
                }
            };
            let name = field
                .ident
                .as_ref()
                .expect("only named field allowed")
                .to_string()
                .to_case(Case::UpperSnake);
            let const_name = format!("{name}_ID");
            let const_name = syn::Ident::new(&const_name, Span::call_site());
            let metadata_name = format!("{name}_METADATA");
            let metadata_name = syn::Ident::new(&metadata_name, Span::call_site());
            quote! {
                #const_name => {
                    static #metadata_name: #conduit::Metadata = #conduit::Metadata {
                        name: std::borrow::Cow::Borrowed(#component_name),
                        component_type: <#ty as #conduit::ConstComponent>::TY,
                        tags: None,
                        asset: false,
                    };
                    Some(&#metadata_name)
                }
            }
        } else {
            quote! {}
        }
    });

    let metadata_items = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        if field.entity_id.is_some() {
            let component_name = match &field.component_id {
                Some(c) => quote! {
                    #c
                },
                None => {
                    quote! {
                        <#ty as #crate_name::conduit::Component>::NAME
                    }
                }
            };
            quote! {
                .chain(std::iter::once(#conduit::Metadata {
                    name: std::borrow::Cow::Borrowed(#component_name),
                    component_type: <#ty as #conduit::ConstComponent>::TY,
                    tags: None,
                    asset: false,
                }))
            }
        } else {
            quote! {
                .chain(<#ty as #crate_name::Metadatatize>::metadata())
            }
        }
    });
    quote! {
        impl #crate_name::Metadatatize for #ident #generics #where_clause {
            fn get_metadata(&self, component_id: #conduit::ComponentId) -> Option<&#conduit::Metadata> {
                use #conduit::ValueRepr;
                #(#ids)*
                match component_id {
                    #(#match_arms)*
                    _ => None
                }
            }

            fn metadata() -> impl Iterator<Item = #conduit::Metadata> {
                std::iter::empty()
                #(#metadata_items)*
            }
        }
    }.into()
}
