use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

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
        let ident = &field.ident.as_ref().expect("field must have ident");
        let component_id = match &field.component_id {
            Some(c) => quote! {
                #impeller::types::ComponentId::new(#c)
            },
            None => {
                quote! {
                    #impeller::types::ComponentId::new(stringify!(#ident))
                }
            }
        };
        if let Some(id) = field.entity_id.or(entity_id) {
            quote! {
                {

                    let schema = <#ty as #impeller::component::Component>::schema();
                    builder.column(
                        #component_id,
                        schema.prim_type(),
                        schema.shape().iter().map(|&s| s as u64),
                        [#impeller::types::EntityId(#id)],
                    )?;
                }
            }
        } else {
            quote! {
                <#ty as #crate_name::AsVTable>::populate_vtable_builder(builder);
            }
        }
    });
    quote! {
        impl #crate_name::AsVTable for #ident #generics #where_clause {
            fn populate_vtable_builder<EntryBuf: #impeller::buf::Buf<#impeller::table::Entry>, DataBuf: #impeller::buf::Buf<u8>>(
                builder: &mut #impeller::table::VTableBuilder<EntryBuf, DataBuf>,
            ) -> Result<(), #impeller::error::Error> {
                #(#vtable_items)*
                Ok(())
            }
        }
    }.into()
}
