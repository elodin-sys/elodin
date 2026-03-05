# Muxide Roadmap

Muxide is a minimal-dependency, pure-Rust MP4 muxer.
Its mission is **simple muxing done right**: encoded frames in, playable MP4 out.

## Current Status: v0.2.0 - Quality & Completeness ✅

### Core Features (v0.1.0)
- ✅ H.264/AVC video muxing (Annex B format)
- ✅ H.265/HEVC video muxing with VPS/SPS/PPS extraction
- ✅ AV1 video muxing (OBU format)
- ✅ VP9 video muxing (complete - frame header parsing, resolution/bit-depth extraction)
- ✅ AAC audio muxing (ADTS format)
- ✅ Opus audio muxing (48kHz raw packets)
- ✅ Fast-start layout (moov before mdat)
- ✅ **Fragmented MP4 for DASH/HLS streaming** (H.264, H.265, AV1, VP9)
- ✅ B-frame support via explicit PTS/DTS
- ✅ Property-based test suite
- ✅ Published to crates.io

### Advanced Features (v0.1.1-0.1.5)
- ✅ **Comprehensive AAC Support**: All profiles (LC, Main, SSR, LTP, HE, HEv2)
- ✅ **World-Class Error Handling**: Detailed diagnostics, hex dumps, JSON output, actionable suggestions
- ✅ **Metadata Support**: Creation time, language encoding (ISO 639-2/T)
- ✅ **API Compatibility**: Builder pattern with fluent API methods
- ✅ **Production Validation**: FFmpeg/ffprobe compatibility verified
- ✅ **Extensive Testing**: 80+ unit tests, property-based tests, 88% coverage
- ✅ **PPT Framework**: Runtime invariant enforcement with 13 contract tests
- ✅ **CI/CD Integration**: Fast unit tests on every commit, comprehensive property tests on PRs
- ✅ **Real-World Examples**: Working demos with fixture data
- ✅ **CLI Tool**: Command-line interface with immediate developer utility
- ✅ **CrabCamera Integration**: Used by production camera plugin (26+ stars)
- ✅ **VP9 Production Readiness**: Full-range support, accurate color metadata
- ✅ **API Modernization**: Unified MuxerBuilder, removed deprecated constructors
- ✅ **Enhanced CLI Diagnostics**: Smart codec detection, better MP4 analysis

## Next Goals (v0.2.0) - Developer Experience & Performance

### v0.1.6 (Q2 2026) - Release Hygiene & Alignment
- [ ] **Git Release Tags**: Add retroactive tags for v0.1.0 through v0.1.5 for better version tracking
- [x] **Dependency Clarity**: Update README.md to clarify "minimal runtime dependencies" (no FFmpeg/C binaries, but Rust crates are used)
- [ ] **Audit Alignment**: Address Muxide-specific findings from CODE_AUDIT_REPORT.md (e.g., API ergonomics) in v0.2.0 planning
- [ ] **CLI Fragmented MP4**: Implement fragmented MP4 support in CLI tool
- [ ] **Documentation Polish**: Complete API docs, better examples, troubleshooting guide

### High Priority
- [ ] **Performance Benchmarks**: Establish baseline performance metrics and optimization targets
- [ ] **SIMD Optimizations**: Performance improvements for hot paths in frame processing

### Medium Priority
- [ ] **Enhanced Documentation**: More real-world examples and tutorials
- [ ] **Async I/O Support**: Optional tokio-based async operations for large file handling
- [ ] **WebAssembly Target**: Browser-based MP4 muxing for web applications

### Lower Priority
- [ ] **Chapter Markers**: Metadata support for navigation points in long videos
- [ ] **Streaming Optimizations**: Further improvements for DASH/HLS low-latency streaming

## Future Possibilities (v0.3.0+)
- [ ] **Hardware-accelerated Muxing**: GPU-assisted frame processing and I/O
- [ ] **Plugin System**: Extensible architecture for custom codecs and formats
- [ ] **Advanced Metadata**: Chapters, subtitles, custom metadata formats
- [ ] **DASH Manifest Generation**: Automatic streaming manifest creation
- [ ] **Cloud Storage Integration**: Direct upload to S3, GCS, Azure Blob Storage
- [ ] **Real-time Streaming**: RTMP/RTSP output for live broadcasting
- [ ] **Container Extensions**: Support for MKV, AVI, MOV formats
- [ ] **Analytics Export**: Performance metrics and usage statistics

## Non-Goals
- **Encoding/decoding** - Muxide is a muxer only, bring your own codec
- **Demuxing/parsing** - We write MP4s, not read them
- **Fixing broken input** - Garbage in, error out
- **Feature bloat** - Every feature must justify its complexity

---

## Recent Achievements
- **v0.1.4 Release**: Production integration with CrabCamera camera plugin
- **Codebase Cleanup**: Removed all external crate references, focused on Muxide's unique value
- **Quality Assurance**: Comprehensive testing suite with real-world validation
- **Developer Experience**: Detailed error messages that make debugging 10x faster
- **Ecosystem Growth**: Adopted by production applications with 26+ GitHub stars

## Governance
- **Lead Maintainer:** Michael A. Kuykendall
- Contributions are welcome via Pull Requests
- The roadmap is set by the lead maintainer to preserve project vision
- All PRs require maintainer review and approval
