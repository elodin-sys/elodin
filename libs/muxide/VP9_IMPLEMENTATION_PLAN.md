# VP9 Video Codec Support Implementation Plan

## Overview
Add VP9 video codec support to Muxide, complementing the existing AV1 support. VP9 is a royalty-free video codec developed by Google, commonly used in WebM containers and supported by major browsers.

## Current State Analysis
- ✅ AV1 support exists (OBU format parsing)
- ✅ H.264/H.265 support exists (Annex B format)
- ✅ MP4 container structure established
- ✅ Codec abstraction pattern established

## VP9 Technical Requirements

### VP9 Frame Format
VP9 uses IVF (Intra Video Frame) containers for storage, but for MP4 muxing we need:
- **Compressed VP9 frames** (similar to AV1 OBUs but different structure)
- **Frame headers** containing temporal and spatial information
- **Key frame detection** for sync samples
- **Resolution extraction** from sequence headers

### VP9 Bitstream Structure
```
VP9 Frame:
├── Frame Header (variable length)
│   ├── Frame Marker (2 bytes: 0x49, 0x83, 0x42)
│   ├── Profile (2 bits)
│   ├── Show Existing Frame (1 bit)
│   ├── Frame Type (1 bit: 0=key, 1=inter)
│   ├── Show Frame (1 bit)
│   ├── Error Resilient (1 bit)
│   └── ... (additional header fields)
├── Compressed Data (variable length)
└── Optional: Frame Size (4 bytes)
```

### Implementation Scope

#### Phase 1: Core VP9 Parsing
- [ ] VP9 frame header parsing
- [ ] Key frame detection
- [ ] Resolution extraction from sequence parameters
- [ ] Basic frame validation

#### Phase 2: MP4 Integration
- [ ] VP9 codec configuration box (vpcC)
- [ ] Sample entry creation
- [ ] Frame data extraction and writing
- [ ] Sync sample detection

#### Phase 3: Testing & Validation
- [ ] Unit tests with VP9 fixtures
- [ ] Integration tests
- [ ] FFmpeg compatibility verification
- [ ] Performance benchmarking

## Files to Modify

### Core Implementation
- `src/codec/mod.rs` - Add VP9 module
- `src/codec/vp9.rs` - New VP9 codec implementation
- `src/api.rs` - Add VideoCodec::Vp9 variant
- `src/muxer/mp4.rs` - Integrate VP9 codec handling

### Testing
- `tests/` - Add VP9 test fixtures
- `tests/vp9_muxing.rs` - VP9-specific tests

## Dependencies
- No new external dependencies (maintain minimal-dependency goal)
- Use existing bit manipulation utilities

## Success Criteria
- [ ] VP9 video files mux correctly into MP4
- [ ] FFmpeg can play resulting MP4 files
- [ ] Performance comparable to other codecs
- [ ] Comprehensive test coverage
- [ ] Documentation updated

## Risk Assessment
- **Low Risk**: Similar to AV1 implementation pattern
- **Medium Complexity**: VP9 headers more complex than H.264 but similar to AV1
- **Testing**: Need VP9 sample data (can generate with ffmpeg)

## Timeline Estimate
- Phase 1: 2-3 days (parsing logic)
- Phase 2: 1-2 days (MP4 integration)
- Phase 3: 1-2 days (testing & validation)
- **Total: 4-7 days**

## Next Steps
1. Research VP9 bitstream specification
2. Obtain/create VP9 test fixtures
3. Implement basic frame parsing
4. Integrate with MP4 muxer
5. Test and validate</content>
<parameter name="filePath">c:\Users\micha\repos\muxide\VP9_IMPLEMENTATION_PLAN.md