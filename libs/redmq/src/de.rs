use serde::{de, Deserialize};

use crate::Error;

pub fn from_redis<'a, T: Deserialize<'a>>(
    map: Vec<(String, String)>,
) -> Result<T, redis::RedisError> {
    let mut deserializer = Deserializer {
        input: map,
        next_value: None,
    };
    T::deserialize(&mut deserializer).map_err(|err| {
        redis::RedisError::from((redis::ErrorKind::TypeError, "invalid type", err.to_string()))
    })
}

struct Deserializer {
    input: Vec<(String, String)>,
    next_value: Option<String>,
}

struct StringDeserializer {
    input: String,
}

macro_rules! unsupported_de {
    ($de_func:ident) => {
        fn $de_func<V: de::Visitor<'de>>(self, _visitor: V) -> Result<V::Value, Self::Error> {
            Err(Error::UnsupportedType)
        }
    };
    ($de_func:ident, $($argt:ty),+) => {
        fn $de_func<V: de::Visitor<'de>>(self, $(_: $argt),+, _visitor: V) -> Result<V::Value, Self::Error> {
            Err(Error::UnsupportedType)
        }
    };
}

macro_rules! parse_de {
    ($de_func:ident, $visit_func:ident) => {
        fn $de_func<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            let v = self.input.parse().map_err(|_| Error::FailedToParse)?;
            visitor.$visit_func(v)
        }
    };
}

impl<'de> de::Deserializer<'de> for &mut Deserializer {
    type Error = Error;

    fn deserialize_map<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_map(self)
    }

    fn deserialize_struct<V: de::Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_map(visitor)
    }

    unsupported_de!(deserialize_any);
    unsupported_de!(deserialize_bool);
    unsupported_de!(deserialize_i8);
    unsupported_de!(deserialize_i16);
    unsupported_de!(deserialize_i32);
    unsupported_de!(deserialize_i64);
    unsupported_de!(deserialize_u8);
    unsupported_de!(deserialize_u16);
    unsupported_de!(deserialize_u32);
    unsupported_de!(deserialize_u64);
    unsupported_de!(deserialize_f32);
    unsupported_de!(deserialize_f64);
    unsupported_de!(deserialize_char);
    unsupported_de!(deserialize_str);
    unsupported_de!(deserialize_string);
    unsupported_de!(deserialize_bytes);
    unsupported_de!(deserialize_byte_buf);
    unsupported_de!(deserialize_option);
    unsupported_de!(deserialize_unit);
    unsupported_de!(deserialize_seq);
    unsupported_de!(deserialize_identifier);
    unsupported_de!(deserialize_ignored_any);

    unsupported_de!(deserialize_unit_struct, &'static str);
    unsupported_de!(deserialize_newtype_struct, &'static str);
    unsupported_de!(deserialize_tuple, usize);
    unsupported_de!(deserialize_tuple_struct, &'static str, usize);
    unsupported_de!(deserialize_enum, &'static str, &'static [&'static str]);
}

impl<'de> de::MapAccess<'de> for Deserializer {
    type Error = Error;

    fn next_key_seed<K>(&mut self, seed: K) -> Result<Option<K::Value>, Self::Error>
    where
        K: de::DeserializeSeed<'de>,
    {
        match self.input.pop() {
            Some((key, value)) => {
                self.next_value = Some(value);
                let mut string_de = StringDeserializer { input: key };
                seed.deserialize(&mut string_de).map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V>(&mut self, seed: V) -> Result<V::Value, Self::Error>
    where
        V: de::DeserializeSeed<'de>,
    {
        let next_value = self.next_value.take().unwrap();
        seed.deserialize(&mut StringDeserializer { input: next_value })
    }
}

impl<'de> de::Deserializer<'de> for &mut StringDeserializer {
    type Error = Error;

    fn deserialize_str<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_str(&self.input)
    }

    fn deserialize_string<V: de::Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_string(self.input.clone())
    }

    fn deserialize_identifier<V: de::Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    parse_de!(deserialize_bool, visit_bool);
    parse_de!(deserialize_i8, visit_i8);
    parse_de!(deserialize_i16, visit_i16);
    parse_de!(deserialize_i32, visit_i32);
    parse_de!(deserialize_i64, visit_i64);
    parse_de!(deserialize_u8, visit_u8);
    parse_de!(deserialize_u16, visit_u16);
    parse_de!(deserialize_u32, visit_u32);
    parse_de!(deserialize_u64, visit_u64);
    parse_de!(deserialize_f32, visit_f32);
    parse_de!(deserialize_f64, visit_f64);
    parse_de!(deserialize_char, visit_char);

    unsupported_de!(deserialize_any);
    unsupported_de!(deserialize_bytes);
    unsupported_de!(deserialize_byte_buf);
    unsupported_de!(deserialize_option);
    unsupported_de!(deserialize_unit);
    unsupported_de!(deserialize_seq);
    unsupported_de!(deserialize_ignored_any);
    unsupported_de!(deserialize_map);

    unsupported_de!(deserialize_unit_struct, &'static str);
    unsupported_de!(deserialize_newtype_struct, &'static str);
    unsupported_de!(deserialize_tuple, usize);
    unsupported_de!(deserialize_tuple_struct, &'static str, usize);
    unsupported_de!(deserialize_struct, &'static str, &'static [&'static str]);
    unsupported_de!(deserialize_enum, &'static str, &'static [&'static str]);
}
