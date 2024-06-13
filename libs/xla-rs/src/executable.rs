use crate::{BufferArgs, PjRtBuffer, Result, Status};

use cpp::{cpp, cpp_class};

use std::pin::Pin;

cpp! {{
    #include "xla/client/xla_builder.h"
    #include "xla/client/lib/constants.h"
    #include "xla/client/lib/matrix.h"
    #include "xla/statusor.h"
    #include "xla/literal_util.h"
    #include "xla/pjrt/pjrt_api.h"
    #include "xla/pjrt/pjrt_c_api_client.h"
    #include "xla/pjrt/pjrt_client.h"
    #include "xla/pjrt/pjrt_stream_executor_client.h"
    #include "xla/pjrt/tfrt_cpu_pjrt_client.h"
    #include "xla/pjrt/gpu/gpu_helpers.h"
    #include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
    using namespace xla;
}}
cpp_class!(pub unsafe struct PjRtLoadedExecutable as "std::shared_ptr<PjRtLoadedExecutable>");

impl PjRtLoadedExecutable {
    pub(crate) fn is_null(&self) -> bool {
        unsafe {
            cpp!([self as "const std::shared_ptr<PjRtLoadedExecutable>*"] -> bool as "bool" {
                return self == nullptr;
            })
        }
    }

    pub fn execute_buffers(&self, buffers: impl BufferArgs) -> Result<Vec<PjRtBuffer>> {
        let out_status: Pin<&mut Status> = std::pin::pin!(Status::ok());
        let untuple_result = buffers.untuple_result();
        let buffers = buffers.get();
        let mut out = vec![];
        {
            let out_ptr = &mut out;
            unsafe {
                cpp!([self as "const std::shared_ptr<PjRtLoadedExecutable>*", buffers as "std::unique_ptr<std::vector<PjRtBuffer*>>", out_status as "Status*", out_ptr as "void*", untuple_result as "bool"] {
                    ExecuteOptions options;
                    options.untuple_result = untuple_result;
                    auto status = (*self)->Execute(absl::Span(buffers.get(), 1), options);
                    if (status.ok()) {
                        std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> bufs = std::move(status).value();
                        for (auto& replica_bufs : bufs) {
                             for (auto& buf : replica_bufs) {
                                 auto out_buf_ptr = rust!(push_out_buf_loaded_exec [out_ptr : &mut Vec<PjRtBuffer> as "void*"] -> *mut PjRtBuffer as "std::unique_ptr<PjRtBuffer>*" {
                                     out_ptr.push(PjRtBuffer::default());
                                     let i = out_ptr.len() - 1;
                                     let ptr = &mut out_ptr[i];
                                     ptr as *mut PjRtBuffer
                                 });
                                 *out_buf_ptr = std::move(buf);
                             }
                        }
                    }else{
                        *out_status = Status(status.status());
                    }
                })
            };
        }
        out_status.to_result()?;
        Ok(out)
    }
}
