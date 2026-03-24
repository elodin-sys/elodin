use darling::FromDeriveInput;
use darling::ast::{self};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(db), supports(struct_named))]
pub struct AsVTable {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    parent: Option<String>,
}

pub fn as_vtable(input: TokenStream) -> TokenStream {
    let impeller = crate::impeller_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let AsVTable {
        ident,
        generics,
        data,
        parent,
    } = AsVTable::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let fields = data.take_struct().unwrap();

    let ts_field = fields.fields.iter().find(|f| f.timestamp);
    let ts_token = ts_field.map(|f| {
        let ts_ident = &f.ident;
        quote! {
            let ts_source = #impeller::vtable::builder::raw_table(
                core::mem::offset_of!(Self, #ts_ident) as u16,
                #impeller::vtable::builder::field_size!(Self, #ts_ident) as u16
            );
        }
    });

    let vtable_items = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let component_id = field.component_id();
        let component_id =
            if let Some(parent) = &parent {
                format!("{parent}.{component_id}")
            }else {
                component_id.to_string()
            };
        let ident = &field.ident;
        let has_ts = ts_field.is_some();
        if field.timestamp || field.skip {
            return quote! {};
        }
        if !field.nest {
            let inner = quote! {
                #impeller::vtable::builder::schema(
                    schema.prim_type(),
                    schema.dim(),
                    #impeller::vtable::builder::component(#component_id)
                )
            };
            let arg = if has_ts {
                quote! { #impeller::vtable::builder::timestamp(ts_source.clone(), #inner) }
            } else {
                inner
            };
            quote! {
                {
                    let schema = <#ty as #impeller::component::Component>::schema();
                    assert_eq!(schema.size(), #impeller::vtable::builder::field_size!(Self, #ident), "to cast to a vtable each field must be the same size as the component");
                    builder.push(
                        #impeller::vtable::builder::field!(
                            Self::#ident,
                            #arg
                        )
                    );
                }
            }
        } else {
            quote! {
                <#ty as #impeller::vtable::AsVTable>::populate_vtable_builder(builder)?;
            }
        }
    });
    quote! {
        impl #impeller::vtable::AsVTable for #ident #generics #where_clause {
            fn populate_vtable_fields(builder: &mut Vec<#impeller::vtable::builder::FieldBuilder>) -> Result<(), #impeller::error::Error> {
                #ts_token
                #(#vtable_items)*
                Ok(())
            }
        }
    }.into()
}
