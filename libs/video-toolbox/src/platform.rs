use crate::{DecodedFrame, Error, H264Decoder, NalType, Result, find_nal_units};
use objc2::rc::Retained;
use objc2_core_foundation::{CFDictionary, CFNumber, CFString};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime,
    CMVideoFormatDescriptionCreateFromH264ParameterSets,
};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
    CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
};
use objc2_video_toolbox::VTDecodeInfoFlags;
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecompressionOutputCallbackRecord, VTDecompressionSession,
};
use std::ptr::NonNull;
use std::sync::{Arc, mpsc};

struct DecoderRx {
    rx: mpsc::Receiver<Result<DecodedFrame>>,
}

struct DecodedTx {
    tx: mpsc::SyncSender<Result<DecodedFrame>>,
}

/// H264 decoder using VideoToolbox
pub struct VideoToolboxDecoder {
    decompression_session: Option<Retained<VTDecompressionSession>>,
    sps_data: Option<Vec<u8>>,
    pps_data: Option<Vec<u8>>,
    tx: Arc<DecodedTx>,
    rx: DecoderRx,
    frame_data: Vec<u8>,
}

// Callback function for decompression
extern "C-unwind" fn decompress_callback(
    tx: *mut std::ffi::c_void,
    _source_frame_ref_con: *mut std::ffi::c_void,
    status: i32,
    _info_flags: VTDecodeInfoFlags,
    buffer: *mut CVPixelBuffer,
    _presentation_time_stamp: CMTime,
    _presentation_duration: CMTime,
) {
    let tx = unsafe { &*(tx as *const DecodedTx) };

    if status != 0 {
        let _ = tx.tx.send(Err(Error::VideoToolbox(status)));
        return;
    }

    let buffer = NonNull::new(buffer);

    // Process the decoded frame
    if let Some(pixel_buffer) = buffer {
        let width = unsafe { CVPixelBufferGetWidth(pixel_buffer.as_ref()) } as usize;
        let height = unsafe { CVPixelBufferGetHeight(pixel_buffer.as_ref()) } as usize;
        let y_stride =
            unsafe { CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer.as_ref(), 0) } as usize;
        let uv_stride =
            unsafe { CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer.as_ref(), 1) } as usize;

        unsafe {
            CVPixelBufferLockBaseAddress(pixel_buffer.as_ref(), CVPixelBufferLockFlags::ReadOnly)
        };

        let y_ptr = unsafe { CVPixelBufferGetBaseAddressOfPlane(pixel_buffer.as_ref(), 0) };
        let uv_ptr = unsafe { CVPixelBufferGetBaseAddressOfPlane(pixel_buffer.as_ref(), 1) };

        if y_ptr.is_null() || uv_ptr.is_null() {
            unsafe {
                CVPixelBufferUnlockBaseAddress(
                    pixel_buffer.as_ref(),
                    CVPixelBufferLockFlags::ReadOnly,
                )
            };
            let _ = tx.tx.send(Err(Error::DecodeFrame));
            return;
        }

        let y_size = y_stride * height;
        let uv_height = (height + 1) / 2;
        let uv_size = uv_stride * uv_height;

        // ideally we would somehow handle this in a zero copy manner, but the pixel buffers only
        // lives for as long as this callback
        let y_plane = unsafe { std::slice::from_raw_parts(y_ptr as *const u8, y_size).to_vec() };
        let uv_plane = unsafe { std::slice::from_raw_parts(uv_ptr as *const u8, uv_size).to_vec() };

        unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer.as_ref(), CVPixelBufferLockFlags::ReadOnly)
        };

        let frame = DecodedFrame {
            y_plane,
            uv_plane,
            width,
            height,
            y_stride,
            uv_stride,
        };

        let _ = tx.tx.send(Ok(frame));
    } else {
        let _ = tx.tx.send(Err(Error::DecodeFrame));
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

        let tx = Arc::clone(&self.tx);
        let callback = VTDecompressionOutputCallbackRecord {
            decompressionOutputCallback: Some(decompress_callback),
            decompressionOutputRefCon: Arc::into_raw(tx) as *mut _,
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

impl H264Decoder for VideoToolboxDecoder {
    fn new() -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel(2);

        Ok(Self {
            decompression_session: None,
            sps_data: None,
            pps_data: None,
            frame_data: vec![],
            tx: Arc::new(DecodedTx { tx }),
            rx: DecoderRx { rx },
        })
    }

    fn decode_nal(&mut self, nal_data: &[u8], pts: i64) -> Result<Option<DecodedFrame>> {
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

                let frame = self.rx.rx.try_recv().ok();
                if let Some(result) = frame {
                    result.map(Some)
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Option<DecodedFrame>> {
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

    fn reset(&mut self) -> Result<()> {
        if let Some(session) = self.decompression_session.take() {
            unsafe {
                session.invalidate();
            }
        }

        self.sps_data = None;
        self.pps_data = None;

        Ok(())
    }
}
