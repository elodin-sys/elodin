use serde::{ser, Serialize};

use crate::Error;

pub fn to_redis<T: Serialize>(value: &T) -> Vec<(String, String)> {
    let mut serializer = Serializer {
        data: Default::default(),
    };
    value.serialize(&mut serializer).unwrap();
    serializer.data
}

struct Serializer {
    data: Vec<(String, String)>,
}

struct StringSerializer {
    data: String,
}

macro_rules! unsupported_ser {
    ($ser_func:ident) => {
        fn $ser_func(self) -> Result<Self::Ok, Self::Error> {
            Err(Error::UnsupportedType)
        }
    };
    ($ser_func:ident, $ser_type:ty) => {
        fn $ser_func(self, _v: $ser_type) -> Result<Self::Ok, Self::Error> {
            Err(Error::UnsupportedType)
        }
    };
}

macro_rules! ser_to_string {
    ($ser_func:ident, $ser_type:ty) => {
        fn $ser_func(self, v: $ser_type) -> Result<(), Error> {
            self.data = v.to_string();
            Ok(())
        }
    };
}

impl ser::Serializer for &mut Serializer {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = ser::Impossible<(), Error>;
    type SerializeTuple = ser::Impossible<(), Error>;
    type SerializeTupleStruct = ser::Impossible<(), Error>;
    type SerializeTupleVariant = ser::Impossible<(), Error>;
    type SerializeMap = ser::Impossible<(), Error>;
    type SerializeStruct = Self;
    type SerializeStructVariant = ser::Impossible<(), Error>;

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }

    unsupported_ser!(serialize_bool, bool);
    unsupported_ser!(serialize_i8, i8);
    unsupported_ser!(serialize_i16, i16);
    unsupported_ser!(serialize_i32, i32);
    unsupported_ser!(serialize_i64, i64);
    unsupported_ser!(serialize_u8, u8);
    unsupported_ser!(serialize_u16, u16);
    unsupported_ser!(serialize_u32, u32);
    unsupported_ser!(serialize_u64, u64);
    unsupported_ser!(serialize_f32, f32);
    unsupported_ser!(serialize_f64, f64);
    unsupported_ser!(serialize_char, char);
    unsupported_ser!(serialize_str, &str);
    unsupported_ser!(serialize_bytes, &[u8]);
    unsupported_ser!(serialize_unit_struct, &'static str);
    unsupported_ser!(serialize_none);
    unsupported_ser!(serialize_unit);

    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<Self::Ok, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Self::Ok, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        Err(Error::UnsupportedType)
    }
}

impl ser::SerializeStruct for &mut Serializer {
    type Ok = ();
    type Error = Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Error> {
        let mut string_serializer = StringSerializer {
            data: Default::default(),
        };
        value.serialize(&mut string_serializer)?;
        self.data.push((key.to_string(), string_serializer.data));
        Ok(())
    }

    fn end(self) -> Result<(), Error> {
        Ok(())
    }
}

impl ser::Serializer for &mut StringSerializer {
    type Ok = ();
    type Error = Error;

    type SerializeSeq = ser::Impossible<(), Error>;
    type SerializeTuple = ser::Impossible<(), Error>;
    type SerializeTupleStruct = ser::Impossible<(), Error>;
    type SerializeTupleVariant = ser::Impossible<(), Error>;
    type SerializeMap = ser::Impossible<(), Error>;
    type SerializeStruct = ser::Impossible<(), Error>;
    type SerializeStructVariant = ser::Impossible<(), Error>;

    ser_to_string!(serialize_bool, bool);
    ser_to_string!(serialize_i8, i8);
    ser_to_string!(serialize_i16, i16);
    ser_to_string!(serialize_i32, i32);
    ser_to_string!(serialize_i64, i64);
    ser_to_string!(serialize_u8, u8);
    ser_to_string!(serialize_u16, u16);
    ser_to_string!(serialize_u32, u32);
    ser_to_string!(serialize_u64, u64);
    ser_to_string!(serialize_f32, f32);
    ser_to_string!(serialize_f64, f64);
    ser_to_string!(serialize_char, char);
    ser_to_string!(serialize_str, &str);

    unsupported_ser!(serialize_bytes, &[u8]);
    unsupported_ser!(serialize_unit_struct, &'static str);
    unsupported_ser!(serialize_none);
    unsupported_ser!(serialize_unit);

    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<(), Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _value: &T,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<(), Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Error> {
        Err(Error::UnsupportedType)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Error> {
        Err(Error::UnsupportedType)
    }
}
