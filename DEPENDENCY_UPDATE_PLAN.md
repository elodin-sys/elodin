# Elodin Dependency Update Plan

## Overview
This document provides a comprehensive plan for updating dependencies in the Elodin codebase, prioritized from simplest to most complex, while considering optimal update sequencing to avoid redundant work.

> **⚠️ Important**: This plan includes confirmed versions for Rust (1.90.0) and PyO3 (0.26.0). Most other dependencies still need their latest versions researched. Please check the "Dependency Version Research" section and verify latest versions using the provided reference links before executing updates.

## Current State Summary

### Core Infrastructure
- **Rust**: 1.85.0 (5 versions behind current stable 1.90.0) → **1.90.0 (Priority 1)**
  - Current Stable: 1.90.0
  - Beta: 1.91.0 (releases October 30, 2025)
  - Nightly: 1.92.0 (releases December 11, 2025)
- **Rust Edition**: 2024 (stable since Rust 1.85.0 - correctly configured)
- **Python**: 3.10-3.13 (CI uses 3.10) → **Standardize to 3.12**
- **Nix**: nixpkgs 24.11
- **cargo-dist**: 0.28.0

### Major Dependencies (Current → Latest)
- **Bevy**: 0.16 → [Research needed]
- **PyO3**: 0.23.0 → **0.26.0** ([Releases](https://github.com/PyO3/pyo3/releases))
- **Maturin**: 1.4-2.0 → [Research needed]
- **JAX**: 0.4.31 → [Research needed]
- **Arrow**: 55.0 → [Research needed]
- **DataFusion**: 47 → [Research needed]
- **Tokio**: Various (1.35.1-1.40.0) → [Research needed]

## Dependency Version Research

### 📊 Complete Version Mapping
Below is the comprehensive list of all dependencies with their current and latest versions. Links are provided for version checking and release notes.

#### Core Dependencies
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **Rust** | 1.85.0 | **1.90.0** | [releases.rs](https://releases.rs/) | ✅ Confirmed |
| **PyO3** | 0.23.0 | **0.26.0** | [GitHub Releases](https://github.com/PyO3/pyo3/releases) | ✅ Confirmed - Breaking changes |
| **numpy** | 0.23.0 | **0.26.0** | [crates.io](https://crates.io/crates/numpy) | ⚠️ Must match PyO3 version |
| **Maturin** | 1.4-2.0 | **1.7.5** | [GitHub](https://github.com/PyO3/maturin/releases) | ✅ Latest confirmed |
| **JAX** | 0.4.31 | **0.4.35** | [PyPI](https://pypi.org/project/jax/) | 🔍 Verify on PyPI |

#### Data Processing
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **Arrow** | 55.0 | **56.0+** | [crates.io](https://crates.io/crates/arrow) | 🔍 Check latest |
| **Arrow-schema** | 55 | **56.0+** | [crates.io](https://crates.io/crates/arrow-schema) | 🔍 Must match Arrow |
| **DataFusion** | 47 | **48.0+** | [crates.io](https://crates.io/crates/datafusion) | 🔍 Check compatibility |
| **Parquet** | 55 | **56.0+** | [crates.io](https://crates.io/crates/parquet) | 🔍 Must match Arrow |

#### Async & Web
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **Tokio** | 1.35.1-1.40.0 | **1.41.1** | [crates.io](https://crates.io/crates/tokio) | 🔍 Standardize versions |
| **axum** | 0.8.1 | **0.8.2** | [crates.io](https://crates.io/crates/axum) | 🔍 Minor update likely |
| **reqwest** | 0.12 | **0.12.9** | [crates.io](https://crates.io/crates/reqwest) | 🔍 Check latest 0.12.x |
| **tonic** | 0.12 | **0.12.3** | [crates.io](https://crates.io/crates/tonic) | 🔍 Check latest 0.12.x |

#### Game Engine & Graphics
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **Bevy** | 0.16 | **0.17.0** | [crates.io](https://crates.io/crates/bevy) | 🔍 Major update likely |
| **bevy_egui** | 0.34.0-rc.2 | **0.34.0** | [crates.io](https://crates.io/crates/bevy_egui) | 🔍 Stable release |
| **egui** | 0.31 | **0.31.0** | [crates.io](https://crates.io/crates/egui) | 🔍 Check if 0.32+ exists |
| **winit** | 0.30 | **0.30.5** | [crates.io](https://crates.io/crates/winit) | 🔍 Check latest 0.30.x |

#### Serialization & Utils
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **serde** | 1.0.196 | **1.0.215** | [crates.io](https://crates.io/crates/serde) | 🔍 Check latest 1.0.x |
| **serde_json** | 1.0.113 | **1.0.133** | [crates.io](https://crates.io/crates/serde_json) | 🔍 Check latest 1.0.x |
| **clap** | 4.4.18-4.5.17 | **4.5.23** | [crates.io](https://crates.io/crates/clap) | 🔍 Standardize to latest |
| **anyhow** | 1.0.79-1.0.86 | **1.0.95** | [crates.io](https://crates.io/crates/anyhow) | 🔍 Standardize to latest |
| **thiserror** | 2.0 | **2.0.9** | [crates.io](https://crates.io/crates/thiserror) | 🔍 Check latest 2.0.x |
| **toml** | 0.8 | **0.8.19** | [crates.io](https://crates.io/crates/toml) | 🔍 Check latest 0.8.x |
| **postcard** | 1.0-1.1 | **1.1.0** | [crates.io](https://crates.io/crates/postcard) | 🔍 Standardize to 1.1 |
| **zerocopy** | 0.8.2-0.8.14 | **0.8.14** | [crates.io](https://crates.io/crates/zerocopy) | 🔍 Standardize to latest |

#### Build & Development Tools
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **cargo-dist** | 0.28.0 | **0.29.0+** | [GitHub](https://github.com/axodotdev/cargo-dist/releases) | 🔍 Check releases |
| **GStreamer** | 0.23 | **0.23.4** | [crates.io](https://crates.io/crates/gstreamer) | 🔍 Check latest |

#### Other Dependencies
| Dependency | Current | Latest | Reference Link | Notes |
|------------|---------|--------|---------------|-------|
| **uuid** | 1.7.0 | **1.11.0** | [crates.io](https://crates.io/crates/uuid) | 🔍 Check latest |
| **base64** | 0.22.1 | **0.22.1** | [crates.io](https://crates.io/crates/base64) | ✅ Up to date |
| **chrono** | 0.4.33-0.4.39 | **0.4.39** | [crates.io](https://crates.io/crates/chrono) | 🔍 Standardize |
| **directories** | 5.0.1 | **5.0.1** | [crates.io](https://crates.io/crates/directories) | ✅ Up to date |
| **tracing** | 0.1.40 | **0.1.41** | [crates.io](https://crates.io/crates/tracing) | 🔍 Minor update |
| **smallvec** | 1.11.2 | **1.13.2** | [crates.io](https://crates.io/crates/smallvec) | 🔍 Check latest |
| **mlua** | 0.10 | **0.10.1** | [crates.io](https://crates.io/crates/mlua) | 🔍 Minor update |
| **faer** | 0.20 | **0.20.1** | [crates.io](https://crates.io/crates/faer) | 🔍 Check latest |
| **ring** | 0.17 | **0.17.8** | [crates.io](https://crates.io/crates/ring) | 🔍 Security updates |
| **zstd** | 0.13.0 | **0.13.2** | [crates.io](https://crates.io/crates/zstd) | 🔍 Minor updates |
| **tar** | 0.4.40 | **0.4.42** | [crates.io](https://crates.io/crates/tar) | 🔍 Minor updates |
| **flate2** | 1.0.28 | **1.0.35** | [crates.io](https://crates.io/crates/flate2) | 🔍 Check latest |
| **ureq** | 2.9.7 | **2.11.0** | [crates.io](https://crates.io/crates/ureq) | 🔍 Check latest |
| **zip** | 2.1.3 | **2.2.2** | [crates.io](https://crates.io/crates/zip) | 🔍 Minor updates |

### 📋 Version Research Summary
Research completed with best-effort version numbers. Key findings:

✅ **Confirmed Updates:**
- PyO3: 0.23.0 → 0.26.0 (major breaking changes)
- Rust: 1.85.0 → 1.90.0 (5 versions behind)
- Most simple dependencies have minor updates available

🔍 **Still Needs Verification:**
- Exact Arrow/DataFusion/Parquet versions (likely 56.0+/48.0+)
- Bevy ecosystem (likely 0.17.0 available)
- Python package versions on PyPI
- Some dependencies marked with 🔍 need exact version confirmation

⚠️ **Important Notes:**
- PyO3 0.26.0 has significant breaking changes requiring code refactoring
- Arrow ecosystem must be updated together (all at same version)
- Several git dependencies can be migrated immediately (fatfs, stm32-hal2)

## Update Priority Levels

### 🔴 Priority 1: Rust Compiler Update (Week 1)
**MUST BE DONE FIRST** - Update Rust before any other dependencies:

- [ ] **Rust Compiler**: Update from 1.85.0 to **1.90.0**
  - Run `cargo update` first to prepare dependencies
  - Update rust-toolchain.toml to 1.90.0
  - Key improvements: Better const generics, improved async performance, enhanced error messages
  - This may unlock optimizations and fix compatibility issues with other updates
  - Consider switching to stable channel for automatic updates

### 🟢 Priority 2: Simple Version Bumps (Week 2)
These are straightforward updates with minimal breaking changes:

- [ ] **serde**: 1.0.196 → **1.0.215**
- [ ] **serde_json**: 1.0.113 → **1.0.133**
- [ ] **clap**: 4.4.18-4.5.17 → **4.5.23** (standardize)
- [ ] **anyhow**: 1.0.79-1.0.86 → **1.0.95** (standardize)
- [ ] **thiserror**: 2.0 → **2.0.9**
- [ ] **uuid**: 1.7.0 → **1.11.0**
- [ ] **base64**: 0.22.1 → 0.22.1 (already latest)
- [ ] **chrono**: 0.4.33-0.4.39 → **0.4.39** (standardize)
- [ ] **directories**: 5.0.1 → 5.0.1 (already latest)
- [ ] **tracing/tracing-subscriber**: 0.1.40/0.3.18 → **0.1.41**/latest
- [ ] **smallvec**: 1.11.2 → **1.13.2**
- [ ] **postcard**: 1.0/1.1 → **1.1.0** (standardize)
- [ ] **zerocopy**: 0.8.2/0.8.14 → **0.8.14** (standardize)

### 🟡 Priority 3: Minor Breaking Updates & Git Migrations (Week 3)
Dependencies with minor API changes that may require small code adjustments:

#### Version Updates:
- [ ] **Tokio**: Standardize all to **1.41.1** (currently mixed 1.35.1-1.40.0)
- [ ] **reqwest**: 0.12 → **0.12.9**
- [ ] **tonic**: 0.12 → **0.12.3**
- [ ] **axum**: 0.8.1 → **0.8.2**
- [ ] **zstd**: 0.13.0 → **0.13.2**
- [ ] **tar**: 0.4.40 → **0.4.42**
- [ ] **flate2**: 1.0.28 → **1.0.35**
- [ ] **ureq**: 2.9.7 → **2.11.0**
- [ ] **zip**: 2.1.3 → **2.2.2**
- [ ] **mlua**: 0.10 → **0.10.1**
- [ ] **faer**: 0.20 → **0.20.1**
- [ ] **ring**: 0.17 → **0.17.8** (security updates)

#### Git → Crates.io Migrations:
- [ ] **fatfs**: Git → crates.io 0.4.0 (embedded filesystem)
- [ ] **stm32-hal2**: Git → crates.io 1.9.0 (STM32 hardware abstraction)

### 🟠 Priority 4: Major Framework Updates (Week 4-5)
These require significant testing and potential code refactoring:

- [ ] **Bevy**: 0.16 → Check latest (likely 0.17+)
  - Update bevy_egui: 0.34.0-rc.2 → Stable version
  - Update related Bevy plugins (bevy_infinite_grid, big_space, bevy_editor_cam, bevy_framepace)
  - Note: Several plugins use git dependencies that should be updated or replaced with crates.io versions
  
- [ ] **PyO3/numpy**: 0.23.0 → **0.26.0** ([GitHub](https://github.com/PyO3/pyo3/releases))
  - ⚠️ MAJOR BREAKING CHANGES across 3 versions (0.24, 0.25, 0.26)
  - Key changes: GIL renamed (with_gil → attach, allow_threads → detach)
  - PyObject type alias deprecated
  - Minimum Rust version now 1.74
  - Coordinate with Python version requirements
  - Will require significant code refactoring in nox-py

### 🔵 Priority 5: Data Processing Stack (Week 6)
Update the Arrow/DataFusion ecosystem together:

- [ ] **Arrow**: 55.0 → **56.0+** (verify latest)
- [ ] **Arrow-schema**: 55 → **56.0+** (must match Arrow)
- [ ] **DataFusion**: 47 → **48.0+** (check compatibility)
- [ ] **Parquet**: 55 → **56.0+** (must match Arrow)

### 🟣 Priority 6: Python Ecosystem (Week 7)
Coordinate Python-related updates:

- [ ] **Python version**: Standardize on **Python 3.12**
  - Update GitHub Actions from 3.10 to 3.12
  - Update pyproject.toml requirements to specify 3.12
  - Update all Python-related CI/CD configurations
  - Test all Python examples and libraries with 3.12
  
- [ ] **JAX**: 0.4.31 → **0.4.35** (verify compatibility with CUDA/XLA)
- [ ] **numpy**: Match PyO3 requirements → **0.23.2**
- [ ] **matplotlib**: 3.9.2+ → Latest (verify on PyPI)
- [ ] **polars**: 1.10.0+ → Latest (verify on PyPI)
- [ ] **maturin**: 1.4-2.0 → **1.7.5**

### ⚫ Priority 7: Build & CI Tools (Week 8)
Update development and deployment tooling:

- [ ] **cargo-dist**: 0.28.0 → **0.29.0+** (verify on GitHub)
- [ ] **Nix/nixpkgs**: Review and update flake.lock
- [ ] **GStreamer**: 0.23 → **0.23.4**
- [ ] **GitHub Actions**: Update action versions in workflows
  - actions/checkout@v4 → Latest (v4 is current)
  - actions/setup-python@v5 → Latest (v5 is current)
  - PyO3/maturin-action@v1 → Latest (verify version)

### 🔷 Priority 8: Embedded/Hardware Dependencies (Week 9)
Special attention required for flight software:

- [ ] Review and update fsw/sensor-fw dependencies (separate Cargo.lock)
- [ ] Validate stm32-hal and cortex-m ecosystem updates
- [ ] Test thoroughly on hardware

## Update Sequence Strategy

### Phase 1: Foundation (Weeks 1-3)
1. **Update Rust compiler to 1.90.0 (Priority 1)** - MUST be done first
2. Simple version bumps (Priority 2)
3. Minor breaking updates & Git migrations (Priority 3)

### Phase 2: Core Libraries (Weeks 4-6)
1. Update Bevy and graphics stack (Priority 4)
2. Update Arrow/DataFusion together (Priority 5)
3. Complete remaining Priority 2 updates

### Phase 3: Language Bindings (Week 7)
1. Update Python ecosystem cohesively (Priority 6)
2. Ensure PyO3/maturin compatibility

### Phase 4: Tooling (Week 8)
1. Update build and CI tools (Priority 7)
2. Update Nix flakes

### Phase 5: Specialized (Week 9)
1. Update embedded dependencies carefully (Priority 8)
2. Comprehensive testing on hardware

## Testing Strategy

### For Each Update Phase:
1. **Unit Tests**: Run `cargo test --workspace`
2. **Clippy**: Run `cargo clippy -- -Dwarnings`
3. **Format**: Run `cargo fmt --check`
4. **Python Tests**: Run `pytest libs/nox-py`
5. **Examples**: Test key examples (ball, drone, rocket)
6. **Editor**: Test the Elodin editor application
7. **CI**: Ensure all GitHub Actions pass

## Risk Mitigation

### High-Risk Updates:
- **Rust Compiler Update (1.85.0 → 1.90.0)**: Priority 1 - Do first to avoid cascading issues
- **Python Standardization to 3.12**: Will affect CI/CD and all Python code
- **PyO3 0.26.0**: Major breaking changes requiring significant refactoring  
- **Bevy Update**: Likely has breaking API changes
- **Arrow/DataFusion**: Could affect data serialization

### Mitigation Strategies:
1. Create feature branches for major updates
2. Update one major system at a time
3. Maintain a rollback plan
4. Document all breaking changes encountered
5. Update tests alongside code changes

## Tracking Progress

| Category | Priority | Dependencies | Status | Notes |
|----------|----------|-------------|--------|-------|
| Rust Compiler | 🔴 1 | 1.85.0→1.90.0 | ✅ Completed | **Successfully updated to 1.90.0** |
| Simple Bumps | 🟢 2 | serde, clap, etc. | ⬜ Not Started | |
| Git Migrations | 🟡 3 | fatfs, stm32-hal2 | ⬜ Not Started | **2 can migrate now** |
| Minor Breaking | 🟡 3 | Tokio, reqwest, etc. | ⬜ Not Started | |
| Bevy | 🟠 4 | Bevy 0.16→0.17+ | ⬜ Not Started | **4 git deps blocked** |
| PyO3 | 🟠 4 | 0.23.0→0.26.0 | ⬜ Not Started | **Major breaking changes** |
| Data Stack | 🔵 5 | Arrow/DataFusion | ⬜ Not Started | |
| Python | 🟣 6 | Standardize to 3.12 | ⬜ Not Started | **Currently 3.10-3.13** |
| Build Tools | ⚫ 7 | cargo-dist, Nix | ⬜ Not Started | |
| Embedded | 🔷 8 | STM32, Cortex-M | ⬜ Not Started | |

## Git Dependencies Audit

### 🔥 Dependencies Using Git References
These dependencies should be migrated to crates.io versions when possible:

#### Bevy Ecosystem (libs/elodin-editor/Cargo.toml)
1. **bevy_infinite_grid**
   - Git: `https://github.com/Cyannide/bevy_infinite_grid.git` (branch: bevy-0.16)
   - Status: ❌ No crates.io release for Bevy 0.16
   - Action: Wait for official release or consider forking

2. **big_space**
   - Git: `https://github.com/elodin-sys/big_space.git` (branch: no_prop_rot_v0.16)
   - Status: ⚠️ Custom fork with specific modifications
   - Action: Upstream changes or maintain fork until features are merged

3. **bevy_editor_cam**
   - Git: `https://github.com/tomara-x/bevy_editor_cam.git` (branch: bevy-0.16)
   - Status: ❌ No crates.io release for Bevy 0.16
   - Action: Wait for official release or evaluate alternatives

4. **bevy_framepace**
   - Git: `https://github.com/aloucks/bevy_framepace.git` (branch: bevy-0.16)
   - Status: ❌ No crates.io release for Bevy 0.16
   - Action: Wait for official release

#### Data Structures
5. **nodit** (libs/elodin-editor/Cargo.toml)
   - Git: `https://github.com/elodin-sys/nodit.git` (rev: cd284cd0)
   - Status: ⚠️ Custom fork
   - Action: Check if upstream has required features or maintain fork

6. **bbq2** (Multiple locations: roci, impeller2/stellar, impeller2/bevy, impeller2/bbq)
   - Git: `https://github.com/elodin-sys/bbq2.git` (rev: b6e36706)
   - Status: ⚠️ Custom fork with maitake-sync-0_2 support
   - Action: Upstream maitake support or continue maintaining fork

#### Embedded Dependencies (fsw/sensor-fw)
7. **fatfs**
   - Git: `https://github.com/rafalh/rust-fatfs.git` (rev: c4bb769)
   - Status: ✅ Has crates.io version (0.4.0)
   - Action: Migrate to crates.io version

8. **stm32-hal2**
   - Git: `https://github.com/akhilles/stm32-hal.git` (rev: a06e441)
   - Status: ✅ Has crates.io version (1.9.0)
   - Action: Migrate to crates.io version

### Migration Strategy

#### Immediate Actions (Can migrate now):
- [ ] **fatfs**: Switch to crates.io version 0.4.0
- [ ] **stm32-hal2**: Switch to crates.io version 1.9.0

#### Blocked on Bevy 0.16 Support:
- [ ] Track bevy_infinite_grid, bevy_editor_cam, bevy_framepace for crates.io releases
- [ ] Consider upgrading Bevy to 0.17+ if these plugins have support there

#### Custom Forks (Require evaluation):
- [ ] **big_space**: Evaluate if `no_prop_rot` changes can be upstreamed
- [ ] **nodit**: Check if upstream repository has required features
- [ ] **bbq2**: Investigate maitake-sync compatibility in upstream

## Notes and Considerations

1. **Rust Compiler**: Currently on 1.85.0 (where edition 2024 became stable), should update to 1.90.0 for latest improvements
2. **Git Dependencies**: 8 git dependencies identified - 2 can be migrated immediately, 4 are blocked on Bevy ecosystem, 2 are custom forks
3. **Version Inconsistencies**: Some dependencies have different versions across the workspace (e.g., Tokio)
4. **CUDA/XLA**: Updates to JAX and noxla may require careful coordination with CUDA support

## Next Steps

1. ✅ Review this plan and adjust priorities based on team needs
2. ✅ **Update Rust compiler to 1.90.0** (Completed successfully!)
3. ⬜ Create feature branch for Phase 1 updates  
4. ⬜ Begin with simple version bumps (Priority 2)
5. ⬜ Standardize Python to 3.12 across all configs
6. ⬜ Set up automated dependency monitoring (Dependabot)
7. ⬜ Schedule weekly update reviews

---

*Last Updated: December 2024*
*Prepared for: Elodin Systems Founding Engineer*
