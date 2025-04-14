use darling::FromDeriveInput;
use darling::ast::{self};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Generics, Ident, parse_macro_input};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(roci), supports(struct_named))]
pub struct AsVTable {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), crate::Field>,
    entity_id: Option<u64>,
}

pub fn as_vtable(input: TokenStream) -> TokenStream {
    let crate_name = crate::roci_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let AsVTable {
        ident,
        generics,
        data,
        entity_id,
    } = AsVTable::from_derive_input(&input).unwrap();
    let where_clause = &generics.where_clause;
    let impeller = quote! { #crate_name::impeller2 };
    let fields = data.take_struct().unwrap();
    let vtable_items = fields.fields.iter().map(|field| {
        let ty = &field.ty;
        let component_id = field.component_id();
        let ident = &field.ident;
        if let Some(id) = field.entity_id.or(entity_id) {
            quote! {
                {

                    let schema = <#ty as #impeller::component::Component>::schema();
                    assert_eq!(schema.size(), #impeller::vtable::builder::field_size!(Self, #ident), "to cast to a vtable each field must be the same size as the component");
                    builder.push(
                        #impeller::vtable::builder::field!(
                            Self::#ident,
                            #impeller::vtable::builder::schema(
                                schema.prim_type(),
                                schema.dim(),
                                #impeller::vtable::builder::pair(#id, #component_id)
                            )
                        )
                    );
                }
            }
        } else {
            quote! {
                <#ty as #crate_name::AsVTable>::populate_vtable_builder(builder)?;
            }
        }
    });
    quote! {
        impl #crate_name::AsVTable for #ident #generics #where_clause {
            fn populate_vtable_fields(builder: &mut Vec<#impeller::vtable::builder::FieldBuilder>) -> Result<(), #impeller::error::Error> {
                #(#vtable_items)*
                Ok(())
            }
        }
    }.into()
}
