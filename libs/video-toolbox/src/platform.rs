use crate::{DecodedFrame, Error, NalType, Result, find_nal_units};
use objc2::rc::Retained;
use objc2_core_foundation::{CFDictionary, CFNumber, CFString};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime,
    CMVideoFormatDescriptionCreateFromH264ParameterSets,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetDataSize, CVPixelBufferGetHeight, CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_24RGB,
    kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecompressionOutputCallbackRecord, VTDecompressionSession,
};
use objc2_video_toolbox::{VTDecodeInfoFlags, VTPixelTransferSession};
use std::cell::RefCell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

struct DecompressContext {
    frame: Rc<RefCell<Option<Result<DecodedFrame>>>>,
    pixel_transfer: Retained<VTPixelTransferSession>,
    pub desired_width: Arc<AtomicUsize>,
}

/// H264 decoder using VideoToolbox
pub struct VideoToolboxDecoder {
    decompression_session: Option<Retained<VTDecompressionSession>>,
    sps_data: Option<Vec<u8>>,
    pps_data: Option<Vec<u8>>,
    ctx: Rc<DecompressContext>,
    frame: Rc<RefCell<Option<Result<DecodedFrame>>>>,
    frame_data: Vec<u8>,
}

impl DecompressContext {
    pub fn set_frame(&self, frame: DecodedFrame) {
        let mut out = self.frame.borrow_mut();
        *out = Some(Ok(frame));
    }

    pub fn set_err(&self, err: Error) {
        let mut out = self.frame.borrow_mut();
        *out = Some(Err(err));
    }
}

extern "C-unwind" fn decompress_callback(
    ctx: *mut std::ffi::c_void,
    _source_frame_ref_con: *mut std::ffi::c_void,
    status: i32,
    _info_flags: VTDecodeInfoFlags,
    buffer: *mut CVPixelBuffer,
    _presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    let ctx = unsafe { &*(ctx as *const DecompressContext) };

    if status != 0 {
        ctx.set_err(Error::VideoToolbox(status));
        return;
    }

    let buffer = NonNull::new(buffer);

    if let Some(pixel_buffer) = buffer {
        let in_width = unsafe { CVPixelBufferGetWidth(pixel_buffer.as_ref()) } as f64;
        let in_height = unsafe { CVPixelBufferGetHeight(pixel_buffer.as_ref()) } as f64;

        let desired_width = ctx.desired_width.load(Ordering::Relaxed);
        let aspect_ratio = in_height / in_width;
        let new_width = in_width.min(desired_width as f64);
        let new_height = (new_width * aspect_ratio) as usize;
        let new_width = new_width as usize;

        let mut out_buffer = std::ptr::null_mut();
        let status = unsafe {
            CVPixelBufferCreate(
                None,
                new_width,
                new_height,
                kCVPixelFormatType_24RGB,
                None,
                NonNull::new_unchecked(&mut out_buffer),
            )
        };
        if status != 0 {
            return;
        }

        let Some(out_buffer) = NonNull::new(out_buffer) else {
            ctx.set_err(Error::DecodeFrame);
            return;
        };

        let status = unsafe {
            ctx.pixel_transfer
                .transfer_image(pixel_buffer.as_ref(), out_buffer.as_ref())
        };
        if status != 0 {
            ctx.set_err(Error::DecodeFrame);
            return;
        }

        let width = unsafe { CVPixelBufferGetWidth(out_buffer.as_ref()) } as usize;
        let height = unsafe { CVPixelBufferGetHeight(out_buffer.as_ref()) } as usize;
        let stride = unsafe { CVPixelBufferGetBytesPerRow(out_buffer.as_ref()) } as usize / 3;

        unsafe {
            CVPixelBufferLockBaseAddress(out_buffer.as_ref(), CVPixelBufferLockFlags::ReadOnly)
        };

        let base_addr = unsafe { CVPixelBufferGetBaseAddress(out_buffer.as_ref()) };
        let size = unsafe { CVPixelBufferGetDataSize(out_buffer.as_ref()) };

        if base_addr.is_null() {
            unsafe {
                CVPixelBufferUnlockBaseAddress(
                    out_buffer.as_ref(),
                    CVPixelBufferLockFlags::ReadOnly,
                )
            };

            ctx.set_err(Error::DecodeFrame);
            return;
        }

        let slice = unsafe { std::slice::from_raw_parts(base_addr as *const [u8; 3], size / 3) };
        let mut rgba = Vec::with_capacity(slice.len() * 4);
        for slice in slice.chunks(stride) {
            let slice = &slice[..width];
            for chunk in slice {
                let [r, g, b] = *chunk;
                rgba.extend_from_slice(&[r, g, b, 255]);
            }
        }
        let frame = DecodedFrame {
            rgba,
            width,
            height,
        };

        unsafe {
            CVPixelBufferUnlockBaseAddress(out_buffer.as_ref(), CVPixelBufferLockFlags::ReadOnly)
        };

        ctx.set_frame(frame);
    } else {
        ctx.set_err(Error::DecodeFrame);
    }
}

impl VideoToolboxDecoder {
    fn create_format_description(&self) -> Result<Retained<CMFormatDescription>> {
        let sps = self.sps_data.as_ref().ok_or(Error::MissingParameters)?;
        let pps = self.pps_data.as_ref().ok_or(Error::MissingParameters)?;

        let mut param_sets = unsafe {
            [sps.as_ptr(), pps.as_ptr()].map(|ptr| NonNull::new_unchecked(ptr as *mut _))
        };
        let mut param_set_sizes = [sps.len(), pps.len()];

        unsafe {
            let mut format_description = std::ptr::null();
            let status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
                None,             // allocator
                param_sets.len(), // parameter_set_count
                NonNull::new(param_sets.as_mut_ptr()).unwrap(),
                NonNull::new(param_set_sizes.as_mut_ptr()).unwrap(),
                4, // nal_unit_header_length
                NonNull::new(&mut format_description).unwrap(),
            );

            if status != 0 {
                return Err(Error::FormatDescription);
            }

            Ok(Retained::from_raw(format_description as *mut _).unwrap())
        }
    }

    fn create_decompression_session(&mut self) -> Result<()> {
        // Clear any existing session
        if let Some(session) = self.decompression_session.take() {
            unsafe { session.invalidate() }
        }

        let format_description = self.create_format_description()?;

        let ctx = Rc::clone(&self.ctx);
        let callback = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(decompress_callback),
            decompressionOutputRefCon: Rc::into_raw(ctx) as *mut _,
        };

        let rgba = CFNumber::new_i32(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange as i32);
        let output_dict = unsafe {
            CFDictionary::<CFString, CFNumber>::from_slices(
                &[kCVPixelBufferPixelFormatTypeKey],
                &[rgba.as_ref()],
            )
        };

        let mut session = std::ptr::null_mut();
        let status = unsafe {
            VTDecompressionSession::create(
                None,                // allocator
                &format_description, // video_format_description
                None,
                Some(output_dict.as_opaque()), // destination_image_buffer_attributes
                &callback,                     // output_callback
                NonNull::new_unchecked(&mut session), // decompression_session_out
            )
        };

        if status != 0 {
            return Err(Error::DecompressionSession);
        }

        let session = unsafe { Retained::from_raw(session) };
        self.decompression_session = session;
        Ok(())
    }
}

impl VideoToolboxDecoder {
    pub fn new(desired_width: Arc<AtomicUsize>) -> Result<Self> {
        let frame = Rc::new(RefCell::new(None));
        let pixel_transfer = unsafe {
            let mut session = std::ptr::null_mut();

            let status = VTPixelTransferSession::create(None, NonNull::new_unchecked(&mut session));
            if status != 0 {
                return Err(Error::DecompressionSession);
            }
            Retained::from_raw(session).ok_or(Error::DecompressionSession)?
        };

        Ok(Self {
            decompression_session: None,
            sps_data: None,
            pps_data: None,
            frame_data: vec![],
            ctx: Rc::new(DecompressContext {
                pixel_transfer,
                desired_width,
                frame: frame.clone(),
            }),
            frame,
        })
    }

    pub fn decode_nal(&mut self, nal_data: &[u8], pts: i64) -> Result<Option<DecodedFrame>> {
        if nal_data.is_empty() {
            return Ok(None);
        }

        let nal_type = NalType::from(nal_data[0]);

        match nal_type {
            NalType::Sps => {
                self.sps_data = Some(nal_data.to_vec());
                Ok(None)
            }
            NalType::Pps => {
                self.pps_data = Some(nal_data.to_vec());
                Ok(None)
            }
            NalType::Idr | NalType::Slice => {
                // Ensure we have SPS and PPS data
                if self.sps_data.is_none() || self.pps_data.is_none() {
                    return Err(Error::MissingParameters);
                }

                // Create decompression session if needed
                if self.decompression_session.is_none() {
                    self.create_decompression_session()?;
                }

                let session = self
                    .decompression_session
                    .as_ref()
                    .ok_or(Error::DecompressionSession)?;

                self.frame_data.clear();
                // we are accepting annex-b nal packets, but video-toolbox only works with aac packets
                // which have the length prepended as big endian bytes
                self.frame_data
                    .extend_from_slice(&(nal_data.len() as u32).to_be_bytes());
                self.frame_data.extend_from_slice(nal_data);

                let block_buffer = unsafe {
                    let mut block_buffer_out = std::ptr::null_mut();
                    let status = CMBlockBuffer::create_with_memory_block(
                        None,
                        self.frame_data.as_mut_ptr() as *mut _, // memory_block
                        self.frame_data.len(),                  // block_length
                        None,                                   // block_allocator
                        std::ptr::null_mut(),                   // custom_block_source
                        0,                                      // offset_to_data
                        self.frame_data.len(),                  // data_length
                        0,                                      // flags
                        NonNull::new(&mut block_buffer_out as *mut _).unwrap(),
                    );

                    if status != 0 {
                        return Err(Error::DecodeFrame);
                    }

                    Retained::from_raw(block_buffer_out).unwrap()
                };

                let format_description = self.create_format_description()?;

                let timing_info = unsafe {
                    CMSampleTimingInfo {
                        duration: CMTime::with_epoch(1, 30, 1),
                        presentationTimeStamp: CMTime::with_epoch(pts, 1000, 1),
                        decodeTimeStamp: CMTime::with_epoch(pts, 1000, 1),
                    }
                };

                let sample_buffer = unsafe {
                    let mut sample_buffer_out = std::ptr::null_mut();

                    let sample_sizes = [self.frame_data.len()];

                    let status = CMSampleBuffer::create(
                        None,                      // allocator
                        Some(&*block_buffer),      // data_buffer
                        true,                      // data_ready
                        None,                      // make_data_ready_callback
                        std::ptr::null_mut(),      // make_data_ready_refcon
                        Some(&format_description), // format_description
                        1,                         // num_samples
                        1,                         // num_sample_timing_entries
                        &timing_info,              // sample_timing_array
                        1,                         // num_sample_size_entries
                        sample_sizes.as_ptr(),     // sample_size_array
                        NonNull::new(&mut sample_buffer_out as *mut _).unwrap(),
                    );

                    if status != 0 {
                        return Err(Error::DecodeFrame);
                    }

                    sample_buffer_out
                };

                let flags = VTDecodeFrameFlags::empty();
                let frame_ref_con = std::ptr::null_mut();

                let status = unsafe {
                    session.decode_frame(
                        &*sample_buffer,
                        flags,
                        frame_ref_con,
                        std::ptr::null_mut(),
                    )
                };

                if status != 0 {
                    return Err(Error::DecodeFrame);
                }

                let Some(res) = self.frame.borrow_mut().take() else {
                    return Ok(None);
                };
                res.map(Some)
            }
            _ => Ok(None),
        }
    }

    pub fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Option<DecodedFrame>> {
        let nal_units = find_nal_units(packet);

        let mut result = None;

        for nal_unit in nal_units {
            match self.decode_nal(nal_unit.data, pts) {
                Ok(Some(frame)) => {
                    result = Some(frame);
                }
                Err(e) => return Err(e),
                _ => {}
            }
        }

        Ok(result)
    }
}
