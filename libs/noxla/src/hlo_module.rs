use crate::{Result, Status, XlaComputation};
use cpp::{cpp, cpp_class};

cpp! {{
    #include "xla/service/hlo_parser.h"
    #include <strstream>
    using namespace xla;
}}
cpp_class!(pub unsafe struct HloModuleProto as "HloModuleProto");

impl HloModuleProto {
    fn empty() -> Self {
        unsafe {
            cpp!([] -> HloModuleProto as "HloModuleProto" {
                return HloModuleProto();
            })
        }
    }

    pub fn parse_binary(binary: &[u8]) -> Result<Self> {
        let mut out = HloModuleProto::empty();
        let out_ptr = &mut out;
        //let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let binary_ptr = binary.as_ptr();
        let binary_len = binary.len();
        let status = unsafe {
            cpp!([binary_ptr as "char*", binary_len as "size_t", out_ptr as "HloModuleProto*"] -> Status as "Status" {
                std::string data(binary_ptr, binary_len);
                HloSnapshot proto;
                if (!proto.ParseFromString(data) &&
                        !proto.mutable_hlo()->ParseFromString(data) &&
                        !proto.mutable_hlo()->mutable_hlo_module()->ParseFromString(data)) {
                    return Status(
                        InvalidArgument("Failed to parse input as HLO protobuf binary"));
                }
                auto config_status = HloModule::CreateModuleConfigFromProto(proto.hlo().hlo_module(), {});
                if(!config_status.ok()){
                    return config_status.status();
                }
                auto config = std::move(config_status.value());
                auto status = HloModule::CreateFromProto(proto.hlo().hlo_module(), config);
                if(!status.ok()){
                    return status.status();
                }
                auto hlo_module = std::move(status.value());
                *out_ptr = hlo_module->ToProto();
                return Status();
            })
        };
        status.to_result()?;
        Ok(out)
    }

    pub fn computation(&self) -> XlaComputation {
        unsafe {
            cpp!([self as "HloModuleProto*"] -> XlaComputation as "XlaComputation" {
                return XlaComputation(*self);
            })
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let string = unsafe {
            cpp!([self as "HloModuleProto*"] -> cxx::UniquePtr<cxx::CxxString> as "std::unique_ptr<std::string>" {
                std::string out;
                self->SerializeToString(&out);
                return std::make_unique<std::string>(std::move(out));
            })
        };
        string.as_bytes().to_vec()
    }
}
