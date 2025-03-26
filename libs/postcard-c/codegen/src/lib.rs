use convert_case::Casing;
use miette::IntoDiagnostic;
use minijinja::{Environment, value::ViaDeserialize};
use postcard_schema::schema::owned::{
    OwnedDataModelType, OwnedDataModelVariant, OwnedNamedType, OwnedNamedValue,
};

static CPP_STRUCT_TMPL: &str = include_str!("./struct.cpp.jinja");
static CPP_ENUM_TMPL: &str = include_str!("./enum.cpp.jinja");

pub trait SchemaExt {
    fn to_cpp() -> miette::Result<String>;
}

impl<S: postcard_schema::Schema> SchemaExt for S {
    fn to_cpp() -> miette::Result<String> {
        let owned_ty: OwnedNamedType = S::SCHEMA.into();
        generate_cpp(&owned_ty)
    }
}

pub fn generate_cpp(ty: &OwnedNamedType) -> miette::Result<String> {
    let mut env = Environment::new();
    env.add_template("struct", CPP_STRUCT_TMPL)
        .into_diagnostic()?;
    env.add_template("enum", CPP_ENUM_TMPL).into_diagnostic()?;

    env.add_function("cpp_ty", |ty: ViaDeserialize<OwnedNamedType>| {
        to_cpp_ty(ty.0)
    });

    env.add_function("cpp_encode", |ty: ViaDeserialize<OwnedNamedValue>| {
        cpp_encode(&ty.0)
    });

    env.add_function("cpp_decode", |ty: ViaDeserialize<OwnedNamedValue>| {
        cpp_decode(&ty.0)
    });

    env.add_function("cpp_size", |ty: ViaDeserialize<OwnedNamedValue>| {
        cpp_size(&ty.0)
    });

    env.add_function(
        "cpp_variant_type",
        |variant: ViaDeserialize<OwnedDataModelVariant>| cpp_variant_type(&variant.0),
    );

    env.add_function(
        "cpp_encode_variant",
        |value_name: &str, variant_type: ViaDeserialize<OwnedDataModelVariant>| {
            cpp_encode_variant(value_name, &variant_type.0)
        },
    );

    env.add_function(
        "cpp_decode_variant",
        |value_name: &str, variant_type: ViaDeserialize<OwnedDataModelVariant>| {
            cpp_decode_variant(value_name, &variant_type.0)
        },
    );

    env.add_function(
        "cpp_size_variant",
        |value_name: &str, variant_type: ViaDeserialize<OwnedDataModelVariant>| {
            cpp_size_variant(value_name, &variant_type.0)
        },
    );
    env.add_function(
        "variant_is_unit",
        |variant: ViaDeserialize<OwnedDataModelVariant>| variant_is_unit(&variant.0),
    );
    env.add_filter("snake_case", |value: &str| {
        value.to_case(convert_case::Case::Snake)
    });

    match &ty.ty {
        OwnedDataModelType::Struct(_) => {
            let tmpl = env.get_template("struct").expect("template missing");
            Ok(tmpl.render(ty).into_diagnostic()?)
        }
        OwnedDataModelType::Enum(_) => {
            let tmpl = env.get_template("enum").expect("template missing");
            Ok(tmpl.render(ty).into_diagnostic()?)
        }
        _ => Err(miette::miette!("unsupported data ty")),
    }
}

pub fn to_cpp_ty(named_ty: OwnedNamedType) -> String {
    match named_ty.ty {
        OwnedDataModelType::U8 => "uint8_t".to_string(),
        OwnedDataModelType::U16 => "uint16_t".to_string(),
        OwnedDataModelType::U32 => "uint32_t".to_string(),
        OwnedDataModelType::U64 => "uint64_t".to_string(),
        OwnedDataModelType::I8 => "int8_t".to_string(),
        OwnedDataModelType::I16 => "int16_t".to_string(),
        OwnedDataModelType::I32 => "int32_t".to_string(),
        OwnedDataModelType::I64 => "int64_t".to_string(),
        OwnedDataModelType::F32 => "float".to_string(),
        OwnedDataModelType::F64 => "double".to_string(),
        OwnedDataModelType::Bool => "bool".to_string(),
        OwnedDataModelType::String => "std::string".to_string(),
        OwnedDataModelType::Seq(inner) => format!("std::vector<{}>", to_cpp_ty(*inner)),
        OwnedDataModelType::Map { key, val } => format!(
            "std::unordered_map<{}, {}>",
            to_cpp_ty(*key),
            to_cpp_ty(*val)
        ),
        OwnedDataModelType::Struct(_) => named_ty.name,
        OwnedDataModelType::I128 => "int128_t".to_string(),
        OwnedDataModelType::U128 => "uint128_t".to_string(),
        OwnedDataModelType::Usize => "size_t".to_string(),
        OwnedDataModelType::Isize => "ssize_t".to_string(),
        OwnedDataModelType::Char => "char".to_string(),
        OwnedDataModelType::ByteArray => "std::vector<uint8_t>".to_string(),
        OwnedDataModelType::Option(ty) => format!("std::optional<{}>", to_cpp_ty(*ty)),
        OwnedDataModelType::NewtypeStruct(ty) => to_cpp_ty(*ty.clone()),
        OwnedDataModelType::Enum(_) => named_ty.name,
        OwnedDataModelType::Tuple(tys) => {
            let tys = tys
                .into_iter()
                .map(|ty| to_cpp_ty(ty.clone()))
                .collect::<Vec<_>>()
                .join(", ");
            format!("std::tuple<{}>", tys)
        }
        ty => panic!("unsupported type {:?}", ty),
    }
}

pub fn cpp_size(ty: &OwnedNamedValue) -> String {
    match &ty.ty.ty {
        OwnedDataModelType::U8 => "size += postcard_size_u8();".to_string(),
        OwnedDataModelType::U16 => format!("size += postcard_size_u16({});", ty.name),
        OwnedDataModelType::U32 => format!("size += postcard_size_u32({});", ty.name),
        OwnedDataModelType::U64 => format!("size += postcard_size_u64({});", ty.name),
        OwnedDataModelType::I8 => "size += postcard_size_i8();".to_string(),
        OwnedDataModelType::I16 => format!("size += postcard_size_i16({});", ty.name),
        OwnedDataModelType::I32 => format!("size += postcard_size_i32({});", ty.name),
        OwnedDataModelType::I64 => format!("size += postcard_size_i64({});", ty.name),
        OwnedDataModelType::F32 => "size += postcard_size_f32({});".to_string(),
        OwnedDataModelType::F64 => "size += postcard_size_f64();".to_string(),
        OwnedDataModelType::Bool => "size += postcard_size_bool();".to_string(),
        OwnedDataModelType::String => {
            format!("size += postcard_size_string({}.length());", ty.name)
        }
        OwnedDataModelType::Seq(inner) => {
            let mut out = String::default();
            out += &format!("size += postcard_size_seq({}.size());\n", ty.name);
            out += &format!("for(const auto& val: {}) {{\n  ", ty.name);
            out += &cpp_size(&OwnedNamedValue {
                name: "val".to_string(),
                ty: *inner.clone(),
            })
            .replace("\n", "\n  ");
            out += "\n}";
            out
        }
        OwnedDataModelType::Map { key, val } => {
            let mut out = String::default();
            out += &format!("size += postcard_size_map({}.size());\n", ty.name);
            out += &format!("for(const auto& [k, v]: {}) {{\n  ", ty.name);

            // size for key
            out += &format!(
                "{}\n  ",
                cpp_size(&OwnedNamedValue {
                    name: "k".to_string(),
                    ty: *key.clone(),
                })
                .replace("\n", "\n  ")
            );

            // size for value
            out += &cpp_size(&OwnedNamedValue {
                name: "v".to_string(),
                ty: *val.clone(),
            })
            .replace("\n", "\n  ");

            out += "\n}";
            out
        }
        OwnedDataModelType::Struct(_) => format!("size += {}.encoded_size();", ty.name),
        OwnedDataModelType::ByteArray => {
            format!("size += postcard_size_byte_array({}.size());", ty.name)
        }
        OwnedDataModelType::Option(inner_ty) => {
            format!(
                r#"if({}) {{
                    size += postcard_size_option_some(0);
                    {}
                }}else{{
                    size += postcard_size_option_none();
                }}
                "#,
                ty.name,
                cpp_size(&OwnedNamedValue {
                    name: format!("*{}", ty.name),
                    ty: *inner_ty.clone()
                })
            )
        }
        OwnedDataModelType::NewtypeStruct(inner_ty) => cpp_size(&OwnedNamedValue {
            name: ty.name.clone(),
            ty: *inner_ty.clone(),
        }),
        OwnedDataModelType::Enum(_) => {
            format!("size += {}.encoded_size();", ty.name)
        }
        OwnedDataModelType::Tuple(tys) => {
            let mut out = String::new();
            for (i, inner_ty) in tys.iter().enumerate() {
                out += &format!(
                    r#"{{
                        auto val = std::get<{i}>({});
                        {}
                    }}"#,
                    ty.name,
                    cpp_size(&OwnedNamedValue {
                        name: "val".to_string(),
                        ty: inner_ty.clone()
                    })
                );
            }
            out
        }
        _ => panic!("unsupported type for size"),
    }
}

pub fn cpp_size_variant(value_name: &str, variant: &OwnedDataModelVariant) -> String {
    if let OwnedDataModelVariant::NewtypeVariant(named_ty) = variant {
        cpp_size(&OwnedNamedValue {
            name: value_name.to_string(),
            ty: *named_ty.clone(),
        })
    } else {
        "".to_string()
    }
}

pub fn cpp_encode(ty: &OwnedNamedValue) -> String {
    match &ty.ty.ty {
        OwnedDataModelType::U8 => format!("result = postcard_encode_u8(slice, {});", ty.name),
        OwnedDataModelType::U16 => format!("result = postcard_encode_u16(slice, {});", ty.name),
        OwnedDataModelType::U32 => format!("result = postcard_encode_u32(slice, {});", ty.name),
        OwnedDataModelType::U64 => format!("result = postcard_encode_u64(slice, {});", ty.name),
        OwnedDataModelType::I8 => format!("result = postcard_encode_i8(slice, {});", ty.name),
        OwnedDataModelType::I16 => format!("result = postcard_encode_i16(slice, {});", ty.name),
        OwnedDataModelType::I32 => format!("result = postcard_encode_i32(slice, {});", ty.name),
        OwnedDataModelType::I64 => format!("result = postcard_encode_i64(slice, {});", ty.name),
        OwnedDataModelType::F32 => format!("result = postcard_encode_f32(slice, {});", ty.name),
        OwnedDataModelType::F64 => format!("result = postcard_encode_f64(slice, {});", ty.name),
        OwnedDataModelType::Bool => format!("result = postcard_encode_bool(slice, {});", ty.name),
        OwnedDataModelType::String => {
            format!(
                "result = postcard_encode_string(slice, {}.c_str(), {0}.length());",
                ty.name
            )
        }
        OwnedDataModelType::Seq(inner) => {
            let mut out = String::default();
            out += &format!("result = postcard_start_seq(slice, {}.size());\n", ty.name);
            out += &format!("for(const auto& val: {}) {{\n  ", ty.name);
            out += &cpp_encode(&OwnedNamedValue {
                name: "val".to_string(),
                ty: *inner.clone(),
            })
            .replace("\n", "\n  ");
            out += "\n}";
            out
        }
        OwnedDataModelType::Map { key, val } => {
            let mut out = String::default();
            out += &format!("result = postcard_start_map(slice, {}.size());\n", ty.name);
            out += &format!("for(const auto& [k, v]: {}) {{\n  ", ty.name);

            // encode key
            out += &format!(
                "{}\n  ",
                cpp_encode(&OwnedNamedValue {
                    name: "k".to_string(),
                    ty: *key.clone(),
                })
                .replace("\n", "\n  ")
            );

            // encode value
            out += &cpp_encode(&OwnedNamedValue {
                name: "v".to_string(),
                ty: *val.clone(),
            })
            .replace("\n", "\n  ");

            out += "\n  if(result != POSTCARD_SUCCESS) return result;\n}";
            out
        }
        OwnedDataModelType::Struct(_) => format!("result = {}.encode_raw(slice);", ty.name),
        OwnedDataModelType::I128 => format!("result = postcard_encode_i128(slice, {});", ty.name),
        OwnedDataModelType::U128 => format!("result = postcard_encode_u128(slice, {});", ty.name),
        OwnedDataModelType::Usize => format!("result = postcard_encode_usize(slice, {});", ty.name),
        OwnedDataModelType::Isize => format!("result = postcard_encode_isize(slice, {});", ty.name),
        OwnedDataModelType::Char => format!("result = postcard_encode_char(slice, {});", ty.name),
        OwnedDataModelType::ByteArray => format!(
            "result = postcard_encode_byte_array(slice, {0}.data(), {0}.size());",
            ty.name
        ),
        OwnedDataModelType::Option(inner_ty) => {
            format!(
                r#"if({}) {{
                    result = postcard_encode_option_some(slice); if(result != POSTCARD_SUCCESS) return result;
                    {}
                }}else{{
                    result = postcard_encode_option_none(slice);
                }}
                "#,
                ty.name,
                cpp_encode(&OwnedNamedValue {
                    name: format!("*{}", ty.name),
                    ty: *inner_ty.clone()
                })
            )
        }
        OwnedDataModelType::NewtypeStruct(inner_ty) => cpp_encode(&OwnedNamedValue {
            name: ty.name.clone(),
            ty: *inner_ty.clone(),
        }),
        OwnedDataModelType::Enum(_) => {
            format!("result = {}.encode_raw(slice);", ty.name)
        }
        OwnedDataModelType::Tuple(tys) => {
            let mut out = String::new();
            for (i, inner_ty) in tys.iter().enumerate() {
                out += &format!(
                    r#"
                    {{
                        auto val = std::get<{i}>({});
                        {}
                    }}
                    "#,
                    ty.name,
                    cpp_encode(&OwnedNamedValue {
                        name: "val".to_string(),
                        ty: inner_ty.clone()
                    })
                );
            }
            out
        }
        _ => panic!("unsupported type"),
    }
}

pub fn cpp_decode(ty: &OwnedNamedValue) -> String {
    match &ty.ty.ty {
        OwnedDataModelType::U8 => format!("result = postcard_decode_u8(slice, &{});", ty.name),
        OwnedDataModelType::U16 => format!("result = postcard_decode_u16(slice, &{});", ty.name),
        OwnedDataModelType::U32 => format!("result = postcard_decode_u32(slice, &{});", ty.name),
        OwnedDataModelType::U64 => format!("result = postcard_decode_u64(slice, &{});", ty.name),
        OwnedDataModelType::I8 => format!("result = postcard_decode_i8(slice, &{});", ty.name),
        OwnedDataModelType::I16 => format!("result = postcard_decode_i16(slice, &{});", ty.name),
        OwnedDataModelType::I32 => format!("result = postcard_decode_i32(slice, &{});", ty.name),
        OwnedDataModelType::I64 => format!("result = postcard_decode_i64(slice, &{});", ty.name),
        OwnedDataModelType::F32 => format!("result = postcard_decode_f32(slice, &{});", ty.name),
        OwnedDataModelType::F64 => format!("result = postcard_decode_f64(slice, &{});", ty.name),
        OwnedDataModelType::Bool => format!("result = postcard_decode_bool(slice, &{});", ty.name),
        OwnedDataModelType::String => {
            let mut out = String::default();
            out += &format!("size_t {}_len;\n", ty.name);
            out += &format!(
                "result = postcard_decode_string_len(slice, &{}_len);\n",
                ty.name
            );
            out += "if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("{}.resize({}_len);\n", ty.name, ty.name);
            out += &format!("if ({}_len > 0) {{\n", ty.name);
            out += &format!(
                "    result = postcard_decode_string(slice, {0}.data(), {0}_len, {0}_len);\n",
                ty.name
            );
            out += "    if (result != POSTCARD_SUCCESS) return result;\n";
            out += "}";
            out
        }
        OwnedDataModelType::Seq(inner) => {
            let mut out = String::default();
            out += &format!("size_t {}_len;\n", ty.name);
            out += &format!(
                "result = postcard_decode_seq_len(slice, &{}_len);\n",
                ty.name
            );
            out += "if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("{}.clear();\n", ty.name);
            out += &format!("{}.reserve({}_len);\n", ty.name, ty.name);
            out += &format!("for(size_t i = 0; i < {}_len; i++) {{\n", ty.name);
            out += &format!(
                "    {} val;\n",
                to_cpp_ty(OwnedNamedType {
                    name: "".to_string(),
                    ty: inner.ty.clone()
                })
            );

            let val_decode = cpp_decode(&OwnedNamedValue {
                name: "val".to_string(),
                ty: *inner.clone(),
            })
            .replace("\n", "\n    ");

            out += &format!("    {}\n", val_decode);
            out += &format!("    {}.push_back(val);\n", ty.name);
            out += "}";
            out
        }
        OwnedDataModelType::Map { key, val } => {
            let mut out = String::default();
            out += &format!("size_t {}_len;\n", ty.name);
            out += &format!(
                "result = postcard_decode_map_len(slice, &{}_len);\n",
                ty.name
            );
            out += "if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("{}.clear();\n", ty.name);

            let key_type = to_cpp_ty(OwnedNamedType {
                name: "".to_string(),
                ty: key.ty.clone(),
            });

            let val_type = to_cpp_ty(OwnedNamedType {
                name: "".to_string(),
                ty: val.ty.clone(),
            });

            out += &format!("for(size_t i = 0; i < {}_len; i++) {{\n", ty.name);
            out += &format!("    {} k;\n", key_type);
            let key_decode = cpp_decode(&OwnedNamedValue {
                name: "k".to_string(),
                ty: *key.clone(),
            })
            .replace("\n", "\n    ");

            out += &format!("    {}\n", key_decode);
            out += "    if (result != POSTCARD_SUCCESS) return result;\n";

            out += &format!("    {} v;\n", val_type);

            let val_decode = cpp_decode(&OwnedNamedValue {
                name: "v".to_string(),
                ty: *val.clone(),
            })
            .replace("\n", "\n    ");

            out += &format!("    {}\n", val_decode);
            out += "    if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("    {}[k] = v;\n", ty.name);
            out += "}";
            out
        }
        OwnedDataModelType::Struct(_) | OwnedDataModelType::Enum(_) => {
            format!("result = {}.decode_raw(slice);", ty.name)
        }
        OwnedDataModelType::I128 => format!("result = postcard_decode_i128(slice, &{});", ty.name),
        OwnedDataModelType::U128 => format!("result = postcard_decode_u128(slice, &{});", ty.name),
        OwnedDataModelType::Usize => {
            format!("result = postcard_decode_usize(slice, &{});", ty.name)
        }
        OwnedDataModelType::Isize => {
            format!("result = postcard_decode_isize(slice, &{});", ty.name)
        }
        OwnedDataModelType::Char => format!("result = postcard_decode_char(slice, &{});", ty.name),
        OwnedDataModelType::ByteArray => {
            let mut out = String::default();
            out += &format!("size_t {}_len;\n", ty.name);
            out += &format!(
                "result = postcard_decode_byte_array_len(slice, &{}_len);\n",
                ty.name
            );
            out += "if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("{}.resize({}_len);\n", ty.name, ty.name);
            out += &format!("if ({}_len > 0) {{\n", ty.name);
            out += &format!(
                "    result = postcard_decode_byte_array(slice, {0}.data(), {0}_len, {0}_len);\n",
                ty.name
            );
            out += "    if (result != POSTCARD_SUCCESS) return result;\n";
            out += "}";
            out
        }
        OwnedDataModelType::Option(inner_ty) => {
            let mut out = String::default();
            out += "{";
            out += "bool is_some;\n";
            out += "result = postcard_decode_option_tag(slice, &is_some);\n";
            out += "if (result != POSTCARD_SUCCESS) return result;\n";
            out += "if (is_some) {\n";

            let inner_type = to_cpp_ty(OwnedNamedType {
                name: "".to_string(),
                ty: inner_ty.ty.clone(),
            });

            out += &format!("    {} val;\n", inner_type);

            let val_decode = cpp_decode(&OwnedNamedValue {
                name: "val".to_string(),
                ty: *inner_ty.clone(),
            })
            .replace("\n", "\n    ");

            out += &format!("    {}\n", val_decode);
            out += "    if (result != POSTCARD_SUCCESS) return result;\n";
            out += &format!("    {} = val;\n", ty.name);
            out += "} else {\n";
            out += &format!("    {} = std::nullopt;\n", ty.name);
            out += "}";
            out += "}";
            out
        }
        OwnedDataModelType::NewtypeStruct(inner_ty) => cpp_decode(&OwnedNamedValue {
            name: ty.name.clone(),
            ty: *inner_ty.clone(),
        }),
        OwnedDataModelType::Tuple(tys) => {
            let mut out = String::new();
            for (i, ty) in tys.iter().enumerate() {
                out += &format!(
                    r#"{} val{i};
                    {}"#,
                    to_cpp_ty(ty.clone()),
                    cpp_encode(&OwnedNamedValue {
                        name: format!("val{i}"),
                        ty: ty.clone()
                    })
                );
            }
            let cpp_tys = tys
                .iter()
                .map(|ty| to_cpp_ty(ty.clone()))
                .collect::<Vec<_>>()
                .join(", ");
            let vals = tys
                .iter()
                .enumerate()
                .map(|(i, _)| format!("val{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            out += &format!("{} =  std::tuple<{}>({});", ty.name, cpp_tys, vals);
            out
        }
        _ => panic!("unsupported type: {:?}", ty.name),
    }
}

/// Helper function to generate code for encoding enum variant data
/// Returns the appropriate C++ type for an enum variant
pub fn cpp_variant_type(variant: &OwnedDataModelVariant) -> String {
    match variant {
        OwnedDataModelVariant::UnitVariant => "std::monostate".to_string(),
        OwnedDataModelVariant::NewtypeVariant(named_ty) => to_cpp_ty(*named_ty.clone()),
        OwnedDataModelVariant::TupleVariant(_) => todo!(),
        OwnedDataModelVariant::StructVariant(_) => todo!(),
    }
}

/// Helper function to generate code for encoding enum variant data
pub fn cpp_encode_variant(value_name: &str, variant: &OwnedDataModelVariant) -> String {
    if let OwnedDataModelVariant::NewtypeVariant(named_ty) = variant {
        cpp_encode(&OwnedNamedValue {
            name: value_name.to_string(),
            ty: *named_ty.clone(),
        })
    } else {
        "".to_string()
    }
}

/// Helper function to generate code for decoding enum variant data
pub fn cpp_decode_variant(value_name: &str, variant: &OwnedDataModelVariant) -> String {
    if let OwnedDataModelVariant::NewtypeVariant(named_ty) = variant {
        cpp_decode(&OwnedNamedValue {
            name: value_name.to_string(),
            ty: *named_ty.clone(),
        })
    } else {
        "".to_string()
    }
}

pub fn variant_is_unit(variant: &OwnedDataModelVariant) -> bool {
    matches!(variant, OwnedDataModelVariant::UnitVariant)
}
