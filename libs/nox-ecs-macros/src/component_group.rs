use darling::ast::{self};
use darling::FromDeriveInput;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Generics, Ident};

#[derive(Debug, FromDeriveInput)]
#[darling(attributes(nox), supports(struct_tuple, struct_named))]
pub struct ComponentGroup {
    ident: Ident,
    generics: Generics,
    data: ast::Data<(), syn::Field>,
}

pub fn component_group(input: TokenStream) -> TokenStream {
    let crate_name = crate::nox_ecs_crate_name();
    let input = parse_macro_input!(input as DeriveInput);
    let ComponentGroup {
        ident,
        generics,
        data,
    } = ComponentGroup::from_derive_input(&input).unwrap();
    let fields = data.take_struct().unwrap();
    let where_clause = &generics.where_clause;
    let params = fields
        .fields
        .iter()
        .map(|f| f.ty.clone())
        .collect::<Vec<_>>();
    quote! {
        impl #generics #crate_name::ComponentGroup for #ident #generics #where_clause {
            type Params = (Self,);
            type Append<B> = (Self, B);
            fn init_params(builder: &mut #crate_name::PipelineBuilder) -> Result<(), #crate_name::Error> {
                <(#(#params,)*)>::init_params(builder)
            }

            fn component_arrays<'a>(
                builder: &'a #crate_name::PipelineBuilder,
            ) -> impl Iterator<Item = #crate_name::ComponentArray<()>> + 'a {
                <(#(#params,)*)>::component_arrays(builder)
            }

            fn component_types() -> impl Iterator<Item = #crate_name::ComponentType> {
                <(#(#params,)*)>::component_types()
            }


            fn component_ids() -> impl Iterator<Item = #crate_name::ComponentId> {
                <(#(#params,)*)>::component_ids()
            }

            fn component_count() -> usize {
                <(#(#params,)*)>::component_count()
            }


            fn map_axes() -> &'static [usize] {
                <(#(#params,)*)>::map_axes()
            }
        }
    }
    .into()
}
